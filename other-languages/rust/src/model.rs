use arrayfire;

use super::corpus::{Corpus, TokenID};

const BACK_WINDOW: usize = 2;
const EMBD_SIZE: usize = 20;
const BATCH_SIZE: usize = 500;
const EPOCHS: usize = 100;
const FRONT_WINDOW: usize = 2;
const LEARNING_RATE: f32 = 0.05;
const SHUFFLE: bool = true;

pub struct Model;

type Array = arrayfire::Array<f32>;

// Create an NxM matrix
fn af_n_by_m(n: u64, m: u64) -> arrayfire::Dim4 {
    arrayfire::Dim4::new(&[n, m, 1, 1])
}

// #vectoraf
fn af_vector(n: u64) -> arrayfire::Dim4 {
    arrayfire::Dim4::new(&[n, 1, 1, 1])
}

// that's one hot array
fn af_one_hot(n: u64, hot_index: u64) -> Array {
    arrayfire::sparse_from_host(
        1,
        n,
        1,
        &[1.0f32],
        &[n as i32],
        &[0],
        arrayfire::SparseFormat::CSR,
    )
}

impl Model {
    pub fn from_corpus(corpus: &Corpus) {
        let fac = f32::sqrt(6.0) / f32::sqrt((EMBD_SIZE + corpus.vocabulary_size()) as f32);

        // Create the weight matrixes with random values
        let mut w1: Array =
            arrayfire::randu(af_n_by_m(corpus.vocabulary_size() as u64, EMBD_SIZE as u64));
        let mut w2: Array =
            (arrayfire::randu::<f32>(af_n_by_m(EMBD_SIZE as u64, corpus.vocabulary_size() as u64))
                - 0.5f32)
                * fac;

        let data: Vec<(TokenID, Vec<TokenID>)> = corpus
            .batch(BACK_WINDOW, FRONT_WINDOW)
            .into_iter()
            .filter(|context| context.len() > 0)
            .enumerate()
            .collect();

        for epoch in 0..EPOCHS {
            // TODO shuffle data
            dbg!(epoch);

            for batch in data.chunks(BATCH_SIZE) {
                let mut dedw1_batch: Array = arrayfire::constant(0f32, w1.dims());
                let mut dedw2_batch: Array = arrayfire::constant(0f32, w2.dims());

                for (target_word, context_words) in batch {
                    let (dedw1_sample, dedw2_sample) =
                        Self::cbow(corpus, *target_word, &context_words, (&w1, &w2));
                    dedw1_batch += dedw1_sample;
                    dedw2_batch += dedw2_sample;
                }

                w1 -= dedw1_batch * LEARNING_RATE;
                w2 -= dedw2_batch * LEARNING_RATE;
            }
        }
    }

    fn cbow(
        corpus: &Corpus,
        word: TokenID,
        context_words: &[TokenID],
        (w1, w2): (&Array, &Array),
    ) -> (Array, Array) {
        // The input to the NN
        let target_vector: Array = af_one_hot(corpus.vocabulary_size() as u64, word as u64);

        // The expected prediction
        let mut context_vector_data = vec![0f32; corpus.vocabulary_size()];
        let vec_avg = 1.0 / context_words.len() as f32;
        context_words
            .iter()
            .for_each(|ctx| context_vector_data[*ctx] = vec_avg);
        let context_vector: Array = Array::new(
            &context_vector_data,
            af_vector(corpus.vocabulary_size() as u64),
        );

        // calculate hidden activations
        let hidden_activations = arrayfire::matmul(
            &w1,
            &context_vector,
            arrayfire::MatProp::TRANS,
            arrayfire::MatProp::NONE,
        );

        // Calculate output vector via a softmax
        let uu = arrayfire::matmul(
            &w2,
            &hidden_activations,
            arrayfire::MatProp::TRANS,
            arrayfire::MatProp::NONE,
        );

        // softmax
        let expu = arrayfire::exp(&uu);
        let softmax = (&expu / arrayfire::sum_all(&expu).0).cast::<f32>();

        // Calculate output error
        let dedu = softmax - target_vector;
        let dedw2 = arrayfire::matmul(
            &hidden_activations,
            &dedu,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::TRANS,
        );

        // Calculate update to W1
        let dedh = arrayfire::matmul(
            &w2,
            &dedu,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        );
        let dedw1 = arrayfire::matmul(
            &context_vector,
            &dedh,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::TRANS,
        );

        return (dedw1, dedw2);
    }
}
