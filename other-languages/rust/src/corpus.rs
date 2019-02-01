use regex::Regex;

pub type TokenID = usize;
pub struct Corpus {
    // TODO optimize these to avoid the copies using some raw pointers
    source_text: Vec<Vec<TokenID>>,
    id_to_token: Vec<String>,
    token_to_id: std::collections::HashMap<String, TokenID>,
}

impl Corpus {
    pub fn from_file_text<P: AsRef<std::path::Path>>(file_path: P) -> std::io::Result<Corpus> {
        let sentence_regex = Regex::new("[.!?\n]+").unwrap();
        let word_regex = Regex::new(
            "(?:\\s+(?:a|an|the|it|is)(?:\\s+|$)|[\"#$%&()*+,-/:;<=>?@\\[\\\\\\]^_`\\{|\\}~\\s]|'s|'re)+",
        )
        .unwrap();
        let file_content = std::fs::read_to_string(file_path)?.to_lowercase();

        Ok(Self::build_corpus(
            sentence_regex
                .split(&file_content)
                .map(|x| x.trim())
                .filter(|x| x.len() > 0)
                .map(|s| word_regex.split(s)),
        ))
    }

    #[cfg(feature = "json")]
    pub fn from_file_json<P: AsRef<std::path::Path>>(file_path: P) -> std::io::Result<Corpus> {
        let reader = std::io::BufReader::new(std::fs::File::open(file_path)?);
        Ok(Self::build_corpus(
            serde_json::from_reader::<_, Vec<Vec<String>>>(reader)
                .map(|data| data.into_iter().map(|s| s.into_iter()))?,
        ))
    }

    fn build_corpus<I: Iterator<Item = II>, II: Iterator<Item = S>, S>(parsed_text: I) -> Corpus
    where
        S: AsRef<str>,
    {
        let mut id_to_token: Vec<String> = Vec::default();
        let mut token_to_id: std::collections::HashMap<String, TokenID> =
            std::collections::HashMap::default();

        let source_text: Vec<Vec<TokenID>> = parsed_text
            .map(|s| {
                s.map(|w| {
                    let token = token_to_id.get(w.as_ref());
                    if token.is_none() {
                        let id = id_to_token.len();
                        let word = String::from(w.as_ref());
                        id_to_token.push(word.clone());
                        token_to_id.insert(word, id);
                        id
                    } else {
                        *token.unwrap()
                    }
                })
                .collect()
            })
            .collect();

        Corpus {
            source_text,
            id_to_token,
            token_to_id,
        }
    }

    pub fn get_token_by_id(&self, id: TokenID) -> Option<&str> {
        if id < self.id_to_token.len() {
            Some(&self.id_to_token[id])
        } else {
            None
        }
    }

    pub fn get_id_by_token(&self, token: &str) -> Option<TokenID> {
        // We need the map here to make `&TokenID` into `TokenID`, this doesn't cost us anything
        // since `TokenID` is the size of a CPU Word
        self.token_to_id.get(token).map(|x| *x)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn corpus_size(&self) -> usize {
        self.source_text.iter().map(|s| s.len()).sum()
    }

    // Create an array words and their contexts
    pub fn batch(&self, back_window: usize, front_window: usize) -> Vec<Vec<TokenID>> {
        let mut result: Vec<Vec<TokenID>> = (0..self.vocabulary_size())
            .map(|_| Vec::with_capacity(back_window + front_window))
            .collect();
        for sentence in &self.source_text {
            for target_index in 0..sentence.len() {
                let back_index = if target_index >= back_window {
                    target_index - back_window
                } else {
                    0
                };
                let front_index = if target_index + front_window <= sentence.len() {
                    target_index + front_window
                } else {
                    sentence.len()
                };

                let context = &mut result[sentence[target_index]];
                context.extend_from_slice(&sentence[back_index..target_index]);
                context.extend_from_slice(&sentence[target_index..front_index]);
            }
        }

        return result;
    }
}
