// OCR Lite — Copyright (c) 2026 Bernhard Straub
// SPDX-License-Identifier: MIT
//
// The text model's tokenizer from the GGUF metadata (tokenizer.ggml.*), ported from llama.cpp's llama-vocab.cpp +
// unicode.cpp for exactly what this model uses: model "gpt2" (byte-level BPE), pre-tokenizer "qwen2". Any other
// tokenizer is an error, never approximated.
//   encode  special tokens split out first (llama.cpp tokenizer_st_partition with parse_special: the CONTROL /
//           USER_DEFINED / UNKNOWN tokens, longest first) → each raw fragment split by the qwen2 pre-tokenizer
//           (unicode_regex_split_custom_qwen2, hand-written there too: the regex has a lookahead) → bytes mapped to
//           GPT-2's printable code points → BPE merges by rank (lowest first, leftmost on a tie) → ids
//   decode  llama_token_to_piece(special = false): a normal token's code points mapped back to bytes, a USER_DEFINED
//           token's text as is, a CONTROL token nothing
//   eog     llama_vocab_is_eog: eos + the tokens llama.cpp marks by NAME (its list, below) — those also become CONTROL
// Only the prompt pieces are encoded (PREFIX / SUFFIX / the image wrappers); the read itself is only decoded.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};

use crate::gguf::{Gguf, TensorSource, Value};

pub use ocr_decode::sampling::Token;

// llama.cpp llama_token_type (the GGUF tokenizer.ggml.token_type values)
const TYPE_NORMAL: i32 = 1;
const TYPE_UNKNOWN: i32 = 2;
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;
const TYPE_UNUSED: i32 = 5;
const TYPE_BYTE: i32 = 6;

/// llama-vocab.cpp load(): the token texts that end generation (special_eog_ids "maintain a list of tokens that cause
/// end-of-generation", plus the FIM pad / repo / file-separator tokens it also adds) — matched by text, like there
const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
];
const FIM_EOG_TEXTS: &[&str] = &["<|fim_pad|>", "<|fim_repo|>", "<|repo_name|>", "<|file_sep|>"];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Normal,
    Control,
    UserDefined,
    Unknown,
    Other,
}

pub struct Tokenizer {
    tokens: Vec<String>,
    kinds: Vec<Kind>,
    ids: HashMap<String, Token>,
    ranks: HashMap<(String, String), usize>,
    /// CONTROL / USER_DEFINED / UNKNOWN tokens, longest text first (llama.cpp cache_special_tokens)
    specials: Vec<Token>,
    eog: HashSet<Token>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    letter: regex::Regex,
}

fn strings<'a, S: TensorSource>(g: &'a Gguf<S>, key: &str) -> Result<Vec<&'a str>> {
    match g.get(key)? {
        Value::Array(a) => a
            .iter()
            .map(|v| match v {
                Value::Str(s) => Ok(s.as_str()),
                v => bail!("{key} holds {v:?}, not a string"),
            })
            .collect(),
        v => bail!("{key} is {v}, not an array"),
    }
}

fn ints<S: TensorSource>(g: &Gguf<S>, key: &str) -> Result<Vec<i32>> {
    match g.get(key)? {
        Value::Array(a) => a
            .iter()
            .map(|v| match v {
                Value::I32(x) => Ok(*x),
                Value::U32(x) => Ok(*x as i32),
                v => bail!("{key} holds {v:?}, not an integer"),
            })
            .collect(),
        v => bail!("{key} is {v}, not an array"),
    }
}

/// GPT-2's bytes_to_unicode: the printable bytes map to themselves, the others to 256, 257, … in byte order
fn byte_map() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap_or('\0')
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap_or('\0')
        };
    }
    out
}

impl Tokenizer {
    pub fn from_gguf<S: TensorSource>(g: &Gguf<S>) -> Result<Self> {
        let model = g.str("tokenizer.ggml.model")?;
        let pre = g.str("tokenizer.ggml.pre")?;
        if model != "gpt2" || pre != "qwen2" {
            bail!("tokenizer {model} / pre-tokenizer {pre}: this engine implements gpt2 / qwen2 only");
        }
        let tokens: Vec<String> = strings(g, "tokenizer.ggml.tokens")?.into_iter().map(str::to_owned).collect();
        let types = ints(g, "tokenizer.ggml.token_type")?;
        if types.len() != tokens.len() {
            bail!("{} token types for {} tokens", types.len(), tokens.len());
        }
        let mut kinds = Vec::with_capacity(types.len());
        for (i, &t) in types.iter().enumerate() {
            kinds.push(match t {
                TYPE_NORMAL => Kind::Normal,
                TYPE_CONTROL => Kind::Control,
                TYPE_USER_DEFINED => Kind::UserDefined,
                TYPE_UNKNOWN => Kind::Unknown,
                TYPE_UNUSED => Kind::Other,
                TYPE_BYTE => bail!("token {i} is a BYTE token: not a gpt2 vocabulary"),
                t => bail!("token {i} has type {t}"),
            });
        }
        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.insert(t.clone(), i as Token);
        }
        let mut ranks = HashMap::new();
        for (r, m) in strings(g, "tokenizer.ggml.merges")?.into_iter().enumerate() {
            // llama.cpp: the first space from position 1 separates the pair
            let pos = m.char_indices().skip(1).find(|c| c.1 == ' ').map(|c| c.0).with_context(|| format!("merge {r} {m:?} has no space"))?;
            ranks.insert((m[..pos].to_owned(), m[pos + 1..].to_owned()), r);
        }

        let mut eog = HashSet::new();
        eog.insert(g.u32("tokenizer.ggml.eos_token_id")? as Token);
        for text in EOG_TEXTS.iter().chain(FIM_EOG_TEXTS) {
            if let Some(&id) = ids.get(*text) {
                eog.insert(id);
                // llama.cpp: "control-looking token … its type will be overridden"
                kinds[id as usize] = Kind::Control;
            }
        }
        for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if g.has_key(key) {
                eog.insert(g.u32(key)? as Token);
            }
        }
        if eog.iter().any(|&t| t < 0 || t as usize >= tokens.len()) {
            bail!("an end-of-generation id is outside the vocabulary");
        }

        let mut specials: Vec<Token> =
            (0..tokens.len()).filter(|&i| matches!(kinds[i], Kind::Control | Kind::UserDefined | Kind::Unknown)).map(|i| i as Token).collect();
        specials.sort_by_key(|&t| std::cmp::Reverse(tokens[t as usize].len()));

        let byte_to_char = byte_map();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Ok(Self { tokens, kinds, ids, ranks, specials, eog, byte_to_char, char_to_byte, letter: regex::Regex::new(r"^\p{L}$")? })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_eog(&self, t: Token) -> bool {
        self.eog.contains(&t)
    }

    /// the end-of-generation ids, sorted
    pub fn eog_ids(&self) -> Vec<Token> {
        let mut v: Vec<Token> = self.eog.iter().copied().collect();
        v.sort();
        v
    }

    /// the id of one token text (e.g. "<|image_pad|>")
    pub fn id(&self, text: &str) -> Result<Token> {
        self.ids.get(text).copied().with_context(|| format!("{text:?} is not a token"))
    }

    /// llama_tokenize(add_special = false, parse_special = true)
    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        // fragments: Ok(raw text range) or Err(special token)
        let mut frags: Vec<std::result::Result<(usize, usize), Token>> = vec![Ok((0, text.len()))];
        for &sp in &self.specials {
            let st = self.tokens[sp as usize].as_str();
            if st.is_empty() {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len());
            for f in frags {
                match f {
                    Ok((mut a, b)) => {
                        while let Some(m) = text[a..b].find(st) {
                            if m > 0 {
                                next.push(Ok((a, a + m)));
                            }
                            next.push(Err(sp));
                            a += m + st.len();
                        }
                        if a < b {
                            next.push(Ok((a, b)));
                        }
                    }
                    special => next.push(special),
                }
            }
            frags = next;
        }
        let mut out = Vec::new();
        for f in frags {
            match f {
                Err(sp) => out.push(sp),
                Ok((a, b)) => {
                    for word in self.pre_split(&text[a..b]) {
                        self.bpe(word, &mut out)?;
                    }
                }
            }
        }
        Ok(out)
    }

    /// llama_token_to_piece(special = false) as bytes (a token may end inside a UTF-8 character)
    pub fn piece(&self, t: Token, out: &mut Vec<u8>) -> Result<()> {
        let i = usize::try_from(t).ok().filter(|&i| i < self.tokens.len()).with_context(|| format!("token {t} is outside the vocabulary"))?;
        match self.kinds[i] {
            Kind::Normal => {
                for c in self.tokens[i].chars() {
                    out.push(*self.char_to_byte.get(&c).with_context(|| format!("token {t}: {c:?} is no byte code point"))?);
                }
            }
            Kind::UserDefined => out.extend_from_slice(self.tokens[i].as_bytes()),
            Kind::Control | Kind::Unknown | Kind::Other => {}
        }
        Ok(())
    }

    /// unicode_regex_split_custom_qwen2 over one raw fragment: the pre-tokens as sub-slices
    fn pre_split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let cps: Vec<(usize, char)> = text.char_indices().collect();
        let n = cps.len();
        let byte_at = |pos: usize| if pos < n { cps[pos].0 } else { text.len() };
        let cpt = |pos: usize| if pos < n { Some(cps[pos].1) } else { None };
        let letter = |pos: usize| cpt(pos).is_some_and(|c| self.is_letter(c));
        let number = |pos: usize| cpt(pos).is_some_and(|c| c.is_numeric());
        let space = |pos: usize| cpt(pos).is_some_and(|c| c.is_whitespace());
        let mut out = Vec::new();
        let mut prev = 0usize;
        let mut add = |end: usize, out: &mut Vec<&'a str>| {
            if end > prev {
                out.push(&text[byte_at(prev)..byte_at(end)]);
            }
            prev = end;
        };
        let mut pos = 0usize;
        while pos < n {
            let c = cps[pos].1;
            // (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if c == '\'' && pos + 1 < n {
                let c1 = cps[pos + 1].1.to_lowercase().next().unwrap_or('\0');
                if matches!(c1, 's' | 't' | 'm' | 'd') {
                    pos += 2;
                    add(pos, &mut out);
                    continue;
                }
                if pos + 2 < n {
                    let c2 = cps[pos + 2].1.to_lowercase().next().unwrap_or('\0');
                    if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
                        pos += 3;
                        add(pos, &mut out);
                        continue;
                    }
                }
            }
            // [^\r\n\p{L}\p{N}]?\p{L}+
            if !(c == '\r' || c == '\n' || number(pos)) && (letter(pos) || letter(pos + 1)) {
                pos += 1;
                while letter(pos) {
                    pos += 1;
                }
                add(pos, &mut out);
                continue;
            }
            // \p{N}
            if number(pos) {
                pos += 1;
                add(pos, &mut out);
                continue;
            }
            // <space>?[^\s\p{L}\p{N}]+[\r\n]*
            let other = |p: usize| p < n && !(space(p) || letter(p) || number(p));
            let start = if c == ' ' { pos + 1 } else { pos };
            if other(start) {
                pos = start;
                while other(pos) {
                    pos += 1;
                }
                while matches!(cpt(pos), Some('\r' | '\n')) {
                    pos += 1;
                }
                add(pos, &mut out);
                continue;
            }
            let mut n_ws = 0usize;
            let mut last_rn = 0usize;
            while space(pos + n_ws) {
                if matches!(cpt(pos + n_ws), Some('\r' | '\n')) {
                    last_rn = pos + n_ws + 1;
                }
                n_ws += 1;
            }
            // \s*[\r\n]+
            if last_rn > 0 {
                pos = last_rn;
                add(pos, &mut out);
                continue;
            }
            // \s+(?!\S)
            if n_ws > 1 && pos + n_ws < n {
                pos += n_ws - 1;
                add(pos, &mut out);
                continue;
            }
            // \s+
            if n_ws > 0 {
                pos += n_ws;
                add(pos, &mut out);
                continue;
            }
            pos += 1;
            add(pos, &mut out);
        }
        out
    }

    fn is_letter(&self, c: char) -> bool {
        let mut b = [0u8; 4];
        self.letter.is_match(c.encode_utf8(&mut b))
    }

    /// byte-level BPE of one pre-token (llm_tokenizer_bpe_session: lowest merge rank first, leftmost on a tie)
    fn bpe(&self, word: &str, out: &mut Vec<Token>) -> Result<()> {
        let mut syms: Vec<String> = word.bytes().map(|b| self.byte_to_char[b as usize].to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone()))
                    && best.is_none_or(|b| r < b.0)
                {
                    best = Some((r, i));
                }
            }
            let Some((_, i)) = best else { break };
            let right = syms.remove(i + 1);
            syms[i].push_str(&right);
        }
        for s in syms {
            out.push(*self.ids.get(&s).with_context(|| format!("BPE symbol {s:?} is not a token"))?);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the model's text GGUF: these tests need the model files — `OCR_LITE_MODEL_DIR=<folder with text-q8_0.gguf>
    /// cargo test -p ocr-wgpu -- --include-ignored` (app/ci.sh does so when the variable is set)
    fn tok() -> Tokenizer {
        let dir = std::env::var("OCR_LITE_MODEL_DIR").expect("OCR_LITE_MODEL_DIR: the folder holding text-q8_0.gguf");
        let path = std::path::PathBuf::from(dir).join("text-q8_0.gguf");
        let g = Gguf::open(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        Tokenizer::from_gguf(&g).unwrap()
    }

    /// ids from llama.cpp (llama_tokenize add_special = false, parse_special = true)
    #[test]
    #[ignore = "needs the model: OCR_LITE_MODEL_DIR + --include-ignored"]
    fn encode_matches_llama_cpp() {
        let t = tok();
        let cases: &[(&str, &[Token])] = &[
            ("<|im_start|>system<|im_end|>\n<|im_start|>user\n", &[151644, 8948, 151645, 198, 151644, 872, 198]),
            ("<|im_end|>\n<|im_start|>assistant\n", &[151645, 198, 151644, 77091, 198]),
            ("<|image_pad|>", &[151655]),
            ("Hello world, it's 1234 Kardong in Lahaul!", &[9707, 1879, 11, 432, 594, 220, 16, 17, 18, 19, 730, 567, 644, 304, 67442, 4943, 0]),
            (
                "  Grüße  \n\n\taus  Zürich — ½ ß Ǆ x²  ",
                &[220, 94063, 23455, 18611, 11323, 355, 220, 1863, 5186, 713, 1959, 220, 26062, 1683, 253, 220, 131, 226, 856, 29456, 256],
            ),
            ("don't WE'LL I'm <|im_end|>x<|endoftext|>", &[15007, 944, 19677, 6, 4086, 358, 2776, 220, 151645, 87, 151643]),
        ];
        for (s, want) in cases {
            assert_eq!(t.encode(s).unwrap(), *want, "{s:?}");
        }
    }

    #[test]
    #[ignore = "needs the model: OCR_LITE_MODEL_DIR + --include-ignored"]
    fn eog_matches_llama_cpp() {
        assert_eq!(tok().eog_ids(), vec![128247, 151643, 151645, 151662, 151663, 151664]);
    }

    #[test]
    #[ignore = "needs the model: OCR_LITE_MODEL_DIR + --include-ignored"]
    fn decode_round_trips_and_skips_control() {
        let t = tok();
        let s = "  Grüße  \n\n\taus  Zürich — ½ ß Ǆ x²  ";
        let mut b = Vec::new();
        for id in t.encode(s).unwrap() {
            t.piece(id, &mut b).unwrap();
        }
        assert_eq!(String::from_utf8(b).unwrap(), s);
        let mut b = Vec::new();
        t.piece(151645, &mut b).unwrap();
        assert!(b.is_empty());
    }
}
