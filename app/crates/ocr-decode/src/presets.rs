// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// THE decoding presets = app/presets.json (the app's reads and a server-side read with vLLM / Transformers decode
// alike) — compiled in, read-only, validated here. The settings editor adds the user's OWN presets beside them (a
// shipped preset is duplicated, never edited: the shipped file stays the one source of the presets). No fallback: a
// malformed file is an error.

use anyhow::{Result, bail};
use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};

// the file's data part (build.rs: `cliDefault` + `presets`, its prose left out of the binaries)
const SHIPPED: &str = include_str!(concat!(env!("OUT_DIR"), "/presets.json"));

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Preset {
    pub name: String,
    pub temperature: f64,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub repetition_window: usize,
    pub frequency_penalty: f64,
    pub max_new_tokens: usize,
    /// only for scoring a page with a reference transcription (max(floor, min(maxNewTokens, factor × its tokens)));
    /// kept so the file stays one, unused here
    pub gt_token_cap_factor: f64,
}

impl Preset {
    pub fn greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// a preset's checks (name, ranges)
    pub fn validate(&self) -> Result<()> {
        let name_ok = !self.name.is_empty()
            && self.name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
        if !name_ok {
            bail!("preset name '{}': letters, digits, . _ - only", self.name);
        }
        if self.temperature < 0.0
            || !(self.top_p > 0.0 && self.top_p <= 1.0)
            || (self.greedy() && self.top_p != 1.0)
            || self.repetition_penalty <= 0.0
            || self.repetition_window < 1
            || self.frequency_penalty < 0.0
            || self.max_new_tokens < 1
        {
            bail!(
                "preset '{}' has an invalid value (temperature ≥ 0, 0 < topP ≤ 1 and 1 when greedy, rp > 0, window ≥ 1, fp ≥ 0, max ≥ 1)",
                self.name
            );
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct File {
    cli_default: String,
    presets: Vec<Preset>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Presets {
    pub shipped: Vec<Preset>,
    pub default_name: String,
    pub user: Vec<Preset>,
}

impl Presets {
    /// the shipped presets + the user's (`user_json` = the saved user list, None = none saved yet)
    pub fn load(user_json: Option<&str>) -> Result<Self> {
        let f: File = serde_json::from_str(SHIPPED)?;
        let user: Vec<Preset> = match user_json {
            Some(s) => serde_json::from_str(s)?,
            None => Vec::new(),
        };
        let p = Self { shipped: f.presets, default_name: f.cli_default, user };
        p.check()?;
        Ok(p)
    }

    /// every preset valid, every name once across shipped + user, the default a shipped preset
    pub fn check(&self) -> Result<()> {
        if self.shipped.is_empty() {
            bail!("presets.json holds no preset");
        }
        let mut seen = std::collections::HashSet::new();
        for p in self.shipped.iter().chain(&self.user) {
            p.validate()?;
            if !seen.insert(p.name.as_str()) {
                bail!("preset name '{}' twice (a user preset must not reuse a shipped name)", p.name);
            }
        }
        if !self.shipped.iter().any(|p| p.name == self.default_name) {
            bail!("cliDefault '{}' is no preset", self.default_name);
        }
        Ok(())
    }

    pub fn named(&self, name: &str) -> Result<&Preset> {
        match self.shipped.iter().chain(&self.user).find(|p| p.name == name) {
            Some(p) => Ok(p),
            None => bail!("no preset '{name}'"),
        }
    }
}

/// the sampled-read seed of a page (presets.json seedRule): int(md5(stem).hexdigest()[:8], 16)
pub fn seed_for_stem(stem: &str) -> u64 {
    let d = Md5::digest(stem.as_bytes());
    u32::from_be_bytes([d[0], d[1], d[2], d[3]]) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_load_and_default() {
        let p = Presets::load(None).unwrap();
        let d = p.named(&p.default_name).unwrap();
        assert_eq!(d.name, "greedy_rp1.1");
        assert_eq!(d.repetition_window, 20);
    }

    #[test]
    fn user_may_not_shadow_shipped() {
        let s = Presets::load(None).unwrap();
        let json = serde_json::to_string(&vec![s.shipped[0].clone()]).unwrap();
        assert!(Presets::load(Some(&json)).is_err());
    }

    #[test]
    fn seed_rule() {
        // python: int(hashlib.md5(b"abc").hexdigest()[:8], 16) = 0x90015098
        assert_eq!(seed_for_stem("abc"), 0x9001_5098);
    }
}
