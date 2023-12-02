Learn_Naive_Bayes_Text(Examples, V)

1. Collect all words and other tokens that occur in Examples.
- Vocabulary <-- All distinct words and other tokens in Examples.
2. Calculate the requiered P(vj) and P(wk | vj) probability terms.
- For each target value vj in V do:
    docsj <-- subset of Examples for which the target value is vj.
    P(vj) <-- |docsj|/|Examples|
    Textj <-- A single document created by concatenating all members of docsj
    n <-- total number of words in Textj (counting duplicate words multiple times)
    for each word wk in Vocabulary
        - nk <-- number of times word wk occurs in Textj
        - P(wk | vj) <-- nk+1/n+|Vocabulary|


Note: k, j means the number of iteration.

Classify_Naive_Bayes_Text(Doc)

- positions <-- all word positions in Doc that contain tokens found in Vocabulary
- Return vNB, where vNB = argmax(Pvj) MULTIPLICATION OF EVERYTHING in i of position P(ai | vj)