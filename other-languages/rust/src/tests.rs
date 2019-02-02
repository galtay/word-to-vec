use arrayfire;

pub fn parse_weights(file_path: &str) -> arrayfire::Array<f32> {
    let txt = std::fs::read_to_string(file_path).unwrap();
    let square_data: Vec<Vec<f32>> = txt
        .split("\n")
        .filter(|s| s.len() > 0)
        .map(|s| s.split(",").map(|x| x.parse::<f32>().unwrap()).collect())
        .collect();
    let mut flat_data = Vec::with_capacity(square_data.len() * square_data[0].len());
    square_data
        .iter()
        .for_each(|x| flat_data.extend_from_slice(&x));

    arrayfire::Array::new(
        &flat_data,
        arrayfire::Dim4::new(&[square_data.len() as u64, square_data[0].len() as u64, 1, 1]),
    )
}

#[test]
fn test() {
    let w1 = parse_weights("../../W1_space_docs_big_idea.csv");
    let w2 = parse_weights("../../W2_space_docs_big_idea.csv");
    let corpus =
        super::corpus::Corpus::from_file_json("../../corpora/space_docs_big_idea/corpus.json")
            .unwrap();

    super::model::Model::from_corpus_with_weights(&corpus, w1, w2);
}
