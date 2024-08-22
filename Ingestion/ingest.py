import pandas as pd
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.csv import partition_csv



def extract_text_and_metadata_from_pdf_document(pdf_path):
    """
    Extracts text and metadata from a pdf document
    :param pdf_path: path to the pdf document
    :return: pandas dataframe with extracted text and metadata
    """
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        hi_res_model_name="yolox",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

    data = []
    for c in elements:
        row = {}
        row['Element_Type'] = type(c).__name__
        row['Filename'] = c.metadata.filename
        row['Date_Modified'] = c.metadata.last_modified
        row['Filetype'] = c.metadata.filetype
        row['Page_Number'] = c.metadata.page_number
        row['Text'] = c.text
        data.append(row)

    df = pd.DataFrame(data)
    return df


def extract_text_and_metadata_from_csv_document(csv_path):
    """
    Extracts text and metadata from a csv document
    :param csv_path: path to the csv document
    :return: pandas dataframe with extracted text and metadata
    """
    elements = partition_csv(
        filename=csv_path,
        infer_table_structure=True,
    )

    data = []
    for c in elements:
        row = {}
        row['Element_Type'] = type(c).__name__
        row['Filename'] = c.metadata.filename
        row['Date_Modified'] = c.metadata.last_modified
        row['Filetype'] = c.metadata.filetype
        row['Page_Number'] = c.metadata.page_number
        row['Text'] = c.text
        data.append(row)

    df = pd.DataFrame(data)
    return df