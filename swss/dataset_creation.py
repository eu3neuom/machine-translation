import os
import pandas as pd
import random

from lxml import etree


REFERENCES_PATH = "data/references/"
SYSTEM_OUTPUTS_PATH = "data/system-outputs/"
RANDOM_SEED = 42
TRAIN_PERCENT = 0.84 # parameter used for Transformers
# TRAIN_PERCENT = 0.54 # parameter used for SWSS


def read_document(path):
    document = {
        "setid": [],
        "docid": [],
        "origlang": [],
        "trglang": [],
        "segid": [],
        "segtext": [],
        "sysid": []
    }

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.parse(path, parser=parser)
    root = tree.getroot()

    for doc in root:
        for p in doc:
            for seg in p:
                document["setid"].append(root.attrib["setid"])
                document["docid"].append(doc.attrib["docid"])
                document["origlang"].append(doc.attrib["origlang"])
                document["trglang"].append(root.attrib["trglang"])
                document["segid"].append(int(seg.attrib["id"]))
                document["segtext"].append(seg.text)
                document["sysid"].append(doc.attrib["sysid"])
    
    return pd.DataFrame(document)


def read_documents_in_folder(folder_path):
    references = None
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        document = read_document(file_path)

        if references is None:
            references = document
        else:
            references = pd.concat([references, document])
    return references


def read_system_outputs(system_outputs_path):
    system_outputs = None
    for folder in os.listdir(system_outputs_path):
        per_language_path = os.path.join(system_outputs_path, folder)
        documents = read_documents_in_folder(per_language_path)

        if system_outputs is None:
            system_outputs = documents
        else:
            system_outputs = pd.concat([system_outputs, documents])
    return system_outputs


def create_dataset(references, system_outputs, scores, sources):
    dataset = {
        "setid": [],
        "docid": [],
        "origlang": [],
        "trglang": [],
        "segid": [],
        "segsource": [],
        "segreference": [],
        "segpredict": [],
        "sysid": [],
        "raw_score": [],
        "z_score": []
    }
    scores_dict = {}
    for _, row in scores.iterrows():
        scores_dict[(row["sysid"], row["docid"], int(row["segid"]))] = (row["raw_score"], row["z_score"])
    sources_dict = {}
    for _, row in sources.iterrows():
        sources_dict[(row["docid"], int(row["segid"]))] = row["segtext"]
        
    for _, row in references.iterrows():
        docid = row["docid"]
        origlang = row["origlang"]
        trglang = row["trglang"]
        segid = row["segid"]

        snip = system_outputs[
            (system_outputs["docid"] == docid) & 
            (system_outputs["origlang"] == origlang) & 
            (system_outputs["segid"] == segid)
        ]

        for _, snip_row in snip.iterrows():
            try:
                raw_score, z_score = scores_dict[(snip_row["sysid"], docid, segid)]
            except:
                continue

            dataset["setid"].append(row["setid"])
            dataset["docid"].append(docid)
            dataset["origlang"].append(origlang)
            dataset["trglang"].append(trglang)
            dataset["segid"].append(segid)
            dataset["segsource"].append(sources_dict[(docid, segid)])
            dataset["segreference"].append(row["segtext"])
            dataset["segpredict"].append(snip_row["segtext"])
            dataset["sysid"].append(snip_row["sysid"])
            dataset["raw_score"].append(raw_score)
            dataset["z_score"].append(z_score)

    return pd.DataFrame(dataset)


def read_scores(path):
    scores = pd.read_csv(path, sep=" ")
    dataset = {
        "sysid": [],
        "docid": [],
        "segid": [],
        "raw_score": [],
        "z_score": []
    }
    for _, row in scores.iterrows():
        dataset["sysid"].append(row["SYS"])
        dataset["docid"].append(row["SEGID"].split("::")[0])
        dataset["segid"].append(row["SEGID"].split("::")[1])
        dataset["raw_score"].append(float(row["RAW.SCR"]))
        dataset["z_score"].append(float(row["Z.SCR"]))
    
    return pd.DataFrame(dataset)


def prepare_texts_for_tree_creation(destination_path, dataframe):
    msg = ""
    for _, row in dataframe.iterrows():
        if row["segtext"] is None:
            continue
        msg += row["segtext"] + "\n\n"
    
    with open(destination_path, "w") as file:
        file.write(msg)


def split_dataset(destination_path, dataset):
    documents = list(set(dataset.docid.tolist()))
    documents.sort()
    random.shuffle(documents)
    
    documents_count = len(documents)
    train_docs_count = int(documents_count * TRAIN_PERCENT)
    test_docs_count = documents_count - train_docs_count
    print(f"Docs counts: [{documents_count}]\tTrain counts: [{train_docs_count}]\t Test counts: [{test_docs_count}]")

    train_docs = documents[:train_docs_count]
    test_docs = documents[train_docs_count:]
    print(f"Len train: [{len(train_docs)}]\tLen test: [{len(test_docs)}]")

    train_dataset = dataset[dataset["docid"].isin(train_docs)]
    test_dataset = dataset[dataset["docid"].isin(test_docs)]
    print(f"Train %: [{train_dataset.shape[0] / dataset.shape[0]}]\tTest %: [{test_dataset.shape[0] / dataset.shape[0]}]\n")

    train_dataset.to_csv(os.path.join(destination_path, "train.csv"), index=False)
    test_dataset.to_csv(os.path.join(destination_path, "test.csv"), index=False)


def main():
    random.seed(RANDOM_SEED)

    references = read_documents_in_folder(REFERENCES_PATH)
    system_outputs = read_system_outputs(SYSTEM_OUTPUTS_PATH)
    sources = read_documents_in_folder("data/sources/")

    en_references = references[references["trglang"] == "en"]
    en_sysout = system_outputs[system_outputs["trglang"] == "en"]
    en_scores = read_scores("data/manual-evaluation/de-en.csv")
    en_source = sources[sources["trglang"] == "en"]
    en_dataset = create_dataset(en_references, en_sysout, en_scores, en_source)
    en_dataset.to_csv("data/en/dataset.csv", index=False)

    de_references = references[references["trglang"] == "de"]
    de_sysout = system_outputs[system_outputs["trglang"] == "de"]
    de_scores = read_scores("data/manual-evaluation/en-de.csv")
    de_source = sources[sources["trglang"] == "de"]
    de_dataset = create_dataset(de_references, de_sysout, de_scores, de_source)
    de_dataset.to_csv("data/de/dataset.csv", index=False)

    prepare_texts_for_tree_creation("data/en/en_refs.txt", en_references)
    prepare_texts_for_tree_creation("data/en/en_sysout.txt", en_sysout)
    prepare_texts_for_tree_creation("data/de/de_refs.txt", de_references)
    prepare_texts_for_tree_creation("data/de/de_sysout.txt", de_sysout)

    split_dataset("data/en/", en_dataset)
    split_dataset("data/de/", de_dataset)


if __name__ == "__main__":
    main()