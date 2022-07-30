
# :information_source: Information-Retrieval

In this project, information retrieval is applied to build search engine that returns relevant documents to user according to a given query.

The program builds a search engine in two stages: 
- Inverted Index built from XML files.
- The user enters a query, The program returns a list of ranked relevant documents from the Inverted Index (built offline) according to the query.


## :arrow_right: Deployment

To create the Inverted Index run:

```python vsm_ir.py create_index [corpus_directory]```
  
To return the relevant documents to the query run:

```python vsm_ir.py query [ranking] [index_path] “<question>”```
  
There are two types of ranking functions available to rank documents: tfidf, bm25

## :framed_picture:	Screenshots

![Project Screenshot](file:///D:/Downloads/Infromation_extraction_sc1.jpg?raw=true "Optional Title")


## :man_technologist:	Authors

- [@TamirSadovsky](https://github.com/TamirSadovsky)
- [@LiriNorkin](https://github.com/LiriNorkin)

