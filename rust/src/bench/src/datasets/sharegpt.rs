// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::SampleRequest;
use crate::error::{BenchError, Result};
use crate::tokenizer::TokenizerKind;

/// Validation bounds for full-history ShareGPT requests.
const MIN_LEN: usize = 1;
const MAX_PROMPT_LEN: usize = 128000;
const MAX_TOTAL_LEN: usize = 150000;

/// Default HuggingFace dataset repo and filename for ShareGPT.
const DEFAULT_SHAREGPT_REPO: &str = "anon8231489123/ShareGPT_Vicuna_unfiltered";
const DEFAULT_SHAREGPT_FILE: &str = "ShareGPT_V3_unfiltered_cleaned_split.json";

/// Download the default ShareGPT dataset from HuggingFace Hub.
/// Uses hf-hub's built-in cache — subsequent calls return the cached path instantly.
pub async fn download_sharegpt_dataset() -> Result<String> {
    tracing::info!(
        repository = DEFAULT_SHAREGPT_REPO,
        file = DEFAULT_SHAREGPT_FILE,
        "downloading ShareGPT dataset"
    );
    let repo = crate::hub::HubRepo::dataset(DEFAULT_SHAREGPT_REPO.to_string())
        .map_err(BenchError::Config)?;
    let path = repo.get(DEFAULT_SHAREGPT_FILE).await.map_err(|e| {
        BenchError::Config(format!(
            "Failed to download ShareGPT dataset from '{DEFAULT_SHAREGPT_REPO}': {e}"
        ))
    })?;
    let path_str = path.to_string_lossy().to_string();
    tracing::info!(dataset = "sharegpt", path = %path_str, "dataset is ready");
    Ok(path_str)
}

/// Load and sample from a ShareGPT-format JSON dataset.
///
/// Mirrors Python's ShareGPTDataset from datasets.py:1230-1313.
pub fn load_sharegpt_dataset(
    tokenizer: &TokenizerKind,
    dataset_path: &str,
    num_requests: usize,
    output_len_override: Option<usize>,
    seed: u64,
    request_id_prefix: &str,
    no_oversample: bool,
    disable_shuffle: bool,
) -> Result<Vec<SampleRequest>> {
    // Load JSON file
    let content = std::fs::read_to_string(dataset_path).map_err(|e| {
        BenchError::Config(format!(
            "Failed to read ShareGPT file '{dataset_path}': {e}"
        ))
    })?;

    let data: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| BenchError::Config(format!("Invalid JSON in ShareGPT file: {e}")))?;

    let entries = data
        .as_array()
        .ok_or_else(|| BenchError::Config("ShareGPT file must contain a JSON array".into()))?;

    // Filter entries with at least 2 conversation turns
    let mut filtered: Vec<&serde_json::Value> = entries
        .iter()
        .filter(|entry| {
            entry
                .get("conversations")
                .and_then(|c| c.as_array())
                .map(|a| a.len() >= 2)
                .unwrap_or(false)
        })
        .collect();

    if filtered.is_empty() {
        return Err(BenchError::Config(
            "No valid entries in ShareGPT file (need at least 2 conversation turns)".into(),
        ));
    }

    // Shuffle (unless disabled)
    let mut rng = StdRng::seed_from_u64(seed);
    if !disable_shuffle {
        filtered.shuffle(&mut rng);
    }

    // Sample requests
    let mut samples = Vec::new();
    let mut ind = 0;

    for entry in &filtered {
        if samples.len() >= num_requests {
            break;
        }

        let conversations = entry["conversations"].as_array().unwrap();
        let Some((messages_json, prompt_text, completion)) = build_sharegpt_messages(conversations)
        else {
            continue;
        };

        // Tokenize the full conversation (all sent turns) and the completion.
        let prompt_ids = tokenizer.encode(&prompt_text, false)?;
        let prompt_len = prompt_ids.len();

        let new_output_len = if let Some(override_len) = output_len_override {
            override_len
        } else {
            let completion_ids = tokenizer.encode(&completion, false)?;
            completion_ids.len()
        };

        // Validate sequence lengths.
        let skip_min_output = output_len_override.is_some();
        if !is_valid_sequence(prompt_len, new_output_len, skip_min_output) {
            continue;
        }

        samples.push(SampleRequest {
            prompt: Arc::from(prompt_text.as_str()),
            prompt_len,
            expected_output_len: new_output_len,
            request_id: Some(format!("{request_id_prefix}{ind}")),
            chat_messages_json: Some(Arc::from(messages_json.as_str())),
            ..Default::default()
        });
        ind += 1;
    }

    // Oversample if dataset is smaller than requested
    if samples.len() < num_requests {
        if no_oversample {
            tracing::info!(
                dataset = "sharegpt",
                samples = samples.len(),
                requested = num_requests,
                "skipping dataset oversampling"
            );
        } else if !samples.is_empty() {
            let needed = num_requests - samples.len();
            let original_len = samples.len();
            for i in 0..needed {
                let mut req = samples[rng.random_range(0..original_len)].clone();
                req.request_id = Some(format!("{request_id_prefix}{}", original_len + i));
                samples.push(req);
            }
            tracing::info!(
                dataset = "sharegpt",
                original_samples = original_len,
                samples = samples.len(),
                "oversampled dataset"
            );
        }
    }

    if samples.is_empty() {
        return Err(BenchError::Config(
            "No valid samples after filtering ShareGPT dataset. \
             Try relaxing constraints or using a larger dataset."
                .into(),
        ));
    }

    Ok(samples)
}

/// Build the full multi-turn OpenAI `messages` array from a ShareGPT conversation,
/// keeping every turn up to and including the **last** human/user turn.
///
/// Returns `(messages_json, prompt_text, completion)`:
/// - `messages_json`: serialized `messages` array (an optional leading `system`
///   turn plus all human/gpt turns through the final human turn), sent verbatim.
/// - `prompt_text`: concatenation of all included turn contents, used only for
///   token accounting (`prompt_len`).
/// - `completion`: the assistant reply that follows the final human turn, if any;
///   used only to estimate the output length and is not sent.
///
/// Returns `None` when the conversation has no human/user turn.
///
/// Role mapping: `system` -> `system`, `human`/`user` -> `user`,
/// `gpt`/`assistant` -> `assistant`. Turns without a recognized `from` role are
/// skipped. When no `from` roles are present at all, falls back to the original
/// positional behavior (`[0]` = user prompt, `[1]` = assistant completion).
fn build_sharegpt_messages(
    conversations: &[serde_json::Value],
) -> Option<(String, String, String)> {
    fn role_of(m: &serde_json::Value) -> Option<&str> {
        m.get("from").and_then(|f| f.as_str())
    }
    fn value_of(m: &serde_json::Value) -> &str {
        m.get("value").and_then(|v| v.as_str()).unwrap_or("")
    }

    // Find the last human/user turn; everything up to it is sent as context.
    let last_human = conversations
        .iter()
        .rposition(|m| matches!(role_of(m), Some("human") | Some("user")));

    let Some(last_human) = last_human else {
        // No `from` roles at all: fall back to positional single-turn behavior.
        if conversations.iter().all(|m| role_of(m).is_none()) && conversations.len() >= 2 {
            let prompt = value_of(&conversations[0]);
            if prompt.is_empty() {
                return None;
            }
            let completion = value_of(&conversations[1]);
            let messages = serde_json::json!([{"role": "user", "content": prompt}]);
            let messages_json = serde_json::to_string(&messages).ok()?;
            return Some((messages_json, prompt.to_string(), completion.to_string()));
        }
        return None;
    };

    let mut messages = Vec::new();
    let mut prompt_text = String::new();
    for m in &conversations[..=last_human] {
        let (api_role, content) = match role_of(m) {
            Some("system") => ("system", value_of(m)),
            Some("human") | Some("user") => ("user", value_of(m)),
            Some("gpt") | Some("assistant") => ("assistant", value_of(m)),
            _ => continue,
        };
        if !prompt_text.is_empty() {
            prompt_text.push('\n');
        }
        prompt_text.push_str(content);
        messages.push(serde_json::json!({"role": api_role, "content": content}));
    }

    if prompt_text.is_empty() {
        return None;
    }

    // Assistant reply after the final human turn, used only for output-length
    // estimation (not sent to the model).
    let completion = conversations[last_human + 1..]
        .iter()
        .find(|m| matches!(role_of(m), Some("gpt") | Some("assistant")))
        .map(value_of)
        .unwrap_or("")
        .to_string();

    let messages_json = serde_json::to_string(&messages).ok()?;
    Some((messages_json, prompt_text, completion))
}

/// Validate a sequence based on prompt and output lengths.
fn is_valid_sequence(
    prompt_len: usize,
    output_len: usize,
    skip_min_output_len_check: bool,
) -> bool {
    if prompt_len < MIN_LEN {
        return false;
    }
    if !skip_min_output_len_check && output_len < MIN_LEN {
        return false;
    }
    if prompt_len > MAX_PROMPT_LEN {
        return false;
    }
    if prompt_len + output_len > MAX_TOTAL_LEN {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_sequence() {
        // Valid
        assert!(is_valid_sequence(100, 50, false));
        // Prompt too short
        assert!(!is_valid_sequence(0, 50, false));
        // Output too short
        assert!(!is_valid_sequence(100, 0, false));
        // Output too short but skip check
        assert!(is_valid_sequence(100, 0, true));
        // Prompt too long
        assert!(!is_valid_sequence(128001, 1, false));
        // Combined too long
        assert!(!is_valid_sequence(128000, 22001, false));
        assert!(is_valid_sequence(1, 1, false));
        assert!(is_valid_sequence(128000, 22000, false));
    }

    #[test]
    fn test_build_sharegpt_messages_with_system() {
        let convs = vec![
            serde_json::json!({"from": "system", "value": "You are helpful"}),
            serde_json::json!({"from": "human", "value": "Hi"}),
            serde_json::json!({"from": "gpt", "value": "Hello!"}),
        ];
        let (json, prompt, completion) = build_sharegpt_messages(&convs).unwrap();
        let msgs: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are helpful");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hi");
        // Only up to the last human is included; the trailing gpt is the completion.
        assert_eq!(msgs.as_array().unwrap().len(), 2);
        assert_eq!(completion, "Hello!");
        assert!(prompt.contains("Hi"));
    }

    #[test]
    fn test_build_sharegpt_messages_multi_turn_keeps_history() {
        let convs = vec![
            serde_json::json!({"from": "human", "value": "Q1"}),
            serde_json::json!({"from": "gpt", "value": "A1"}),
            serde_json::json!({"from": "human", "value": "Q2"}),
            serde_json::json!({"from": "gpt", "value": "A2"}),
        ];
        let (json, _prompt, completion) = build_sharegpt_messages(&convs).unwrap();
        let msgs: serde_json::Value = serde_json::from_str(&json).unwrap();
        // Keep every turn through the last human (Q1, A1, Q2); A2 is the completion.
        assert_eq!(msgs.as_array().unwrap().len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Q1");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "A1");
        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], "Q2");
        assert_eq!(completion, "A2");
    }

    #[test]
    fn test_build_sharegpt_messages_trailing_human_no_completion() {
        let convs = vec![
            serde_json::json!({"from": "human", "value": "Q1"}),
            serde_json::json!({"from": "gpt", "value": "A1"}),
            serde_json::json!({"from": "human", "value": "Q2"}),
        ];
        let (json, _prompt, completion) = build_sharegpt_messages(&convs).unwrap();
        let msgs: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(msgs.as_array().unwrap().len(), 3);
        assert_eq!(msgs[2]["content"], "Q2");
        // No assistant reply after the final human turn.
        assert_eq!(completion, "");
    }

    #[test]
    fn test_build_sharegpt_messages_positional_fallback() {
        // No `from` roles — fall back to single-turn positional [0]/[1].
        let convs = vec![
            serde_json::json!({"value": "prompt text"}),
            serde_json::json!({"value": "completion text"}),
        ];
        let (json, prompt, completion) = build_sharegpt_messages(&convs).unwrap();
        let msgs: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(msgs.as_array().unwrap().len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(prompt, "prompt text");
        assert_eq!(completion, "completion text");
    }

    #[test]
    fn test_build_sharegpt_messages_no_human_returns_none() {
        let convs = vec![
            serde_json::json!({"from": "system", "value": "You are helpful"}),
            serde_json::json!({"from": "gpt", "value": "Hello!"}),
        ];
        assert!(build_sharegpt_messages(&convs).is_none());
    }
}
