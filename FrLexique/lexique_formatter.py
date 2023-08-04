import pandas as pd


def format_lexique(lexique_path):
    df = pd.read_csv(lexique_path, sep='\t')
    df = df[['ortho', 'cgram']]
    # Remove the rows with NaN values
    df = df.dropna()
    # Save the lexique in a csv file
    df.to_csv('../data/lexique/lexique.csv', index=False)
    # Remove cgrams
    df = df[['ortho']]
    # Remove same words
    df = df.drop_duplicates(subset=['ortho'])
    # Save the lexique in a csv file
    df.to_csv('../data/lexique/lexique_unique.csv', index=False)

    return df


if __name__ == '__main__':
    format_lexique('../data/lexique/Lexique383.tsv')
