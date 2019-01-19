#[test]
fn test() {
    let corpus = super::corpus::Corpus::from_file("../../space_docs_big_idea.txt").unwrap();
    super::model::Model::from_corpus(&corpus)
}
