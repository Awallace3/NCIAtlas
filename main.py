import numpy as np
import pandas as pd
from pprint import pprint as pp
import qcelemental as qcel
import glob


z_to_elem = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    11: "Na",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
}

lots = {
    # "HF/aTZ": "",
    # "MP2/aTZ/CP": "",
    # "MP2/a5Z/CP": "",
    "HF/aDZ": "HF/aug-cc-pVDZ/CP",
    # "HF/aTZ": "HF/aug-cc-pVTZ/CP",
    "HF/haTZ": "HF/aug-cc-pVTZ/CP",
    "HF/aQZ": "HF/aug-cc-pVQZ/CP",
    "MP2/aDZ/CP": "MP2/aug-cc-pVDZ/CP",
    "MP2/haTZ/CP": "MP2/aug-cc-pVTZ/CP",
    "MP2/aQZ/CP": "MP2/aug-cc-pVQZ/CP",
    # "CCSD/aDZ/CP": "CCSD/aug-cc-pVDZ/CP",
    # "CCSD/haTZ/CP": "CCSD/aug-cc-pVTZ/CP",
    # "CCSD(T)/aDZ/CP": "CCSD(T)/aug-cc-pVDZ/CP",
    # "CCSD(T)/haTZ/CP": "CCSD(T)/aug-cc-pVTZ/CP",
}


def fragment_molecule(mol: qcel.models.Molecule):
    from psi4.driver.qcdb.bfs import BFS

    frags = BFS(mol.geometry, mol.atomic_numbers, bond_threshold=1.4)
    bond_thresholds = np.arange(1.4, 1.0, -0.01)
    for i in bond_thresholds:
        frags = BFS(mol.geometry, mol.atomic_numbers, bond_threshold=i)
        if len(frags) == 2:
            return frags
    # assert len(frags) == 2, "Expected exactly two fragments in the molecule. Got: {}".format(frags)
    print("Expected exactly two fragments in the molecule. Got: {}".format(frags))
    return


def create_pandas_dfs():
    """
    Assemble all NCI datasets in this repository into individual pandas
    dataframes for every set. Create a column named qcel_molecules that is a
    qcel.models.Molecule object for each row. Have other columns in the dataframe
    be other identifying information or interaction energies for a specific
    level of theory.
    """
    import yaml
    import glob
    import os

    class MyLoader(yaml.SafeLoader):
        pass

    def construct_ruby_object(loader, tag_suffix, node):
        return loader.construct_mapping(node)

    MyLoader.add_multi_constructor("!ruby/object:", construct_ruby_object)

    dataset_files = glob.glob("dataset_definition_files/*.yaml")
    dfs = {}

    for dataset_file in dataset_files:
        print(f"\n{dataset_file}")
        with open(dataset_file, "r") as f:
            data = yaml.load(f, Loader=MyLoader)

        dataset_name = data["description"]["name"]
        items = data["items"]
        alternative_reference = data.get("alternative_reference", {})

        df_data = {
            "name": [],
            "shortname": [],
            "reference_value": [],
            "group": [],
            "tags": [],
            "qcel_molecules": [],
            "processing_error": [],
        }
        for item in items:
            geom_file_parts = item["geometry"].split(":")
            geom_dir = geom_file_parts[0]
            geom_file = geom_file_parts[1] + ".xyz"
            geom_path = os.path.join("geometries", geom_dir, geom_file)

            with open(geom_path, "r") as gf:
                xyz_data = gf.read()

            mol = qcel.models.Molecule.from_data(xyz_data, dtype="xyz")
            not_supported = False
            for i in mol.atomic_numbers:
                if i not in z_to_elem:
                    not_supported = True
                    print(f"Element not supported: {i}")

            if not_supported:
                df_data["name"].append(item["name"])
                df_data["shortname"].append(item["shortname"])
                df_data["reference_value"].append(item["reference_value"])
                df_data["group"].append(item["group"])
                df_data["tags"].append(item["tags"])
                df_data["qcel_molecules"].append(mol)
                df_data["processing_error"].append("ELEMENT NOT SUPPORTED")
                continue

            frags = fragment_molecule(mol)
            if frags is None:
                df_data["processing_error"].append("FRAGMENTATION ERROR")
            else:
                mol_dict = dict(mol)
                mol2_dict = {
                    "symbols": mol_dict["symbols"],
                    "geometry": mol_dict["geometry"],
                    "fragments": [np.array(frags[0]), np.array(frags[1])],
                }
                try:
                    mol = qcel.models.Molecule(**mol2_dict)
                    df_data["processing_error"].append(None)
                except Exception as e:
                    print(e)
                    df_data["processing_error"].append("MOL CONSTRUCTION ERROR")
            df_data["name"].append(item["name"])
            df_data["shortname"].append(item["shortname"])
            df_data["reference_value"].append(item["reference_value"])
            df_data["group"].append(item["group"])
            df_data["tags"].append(item["tags"])
            df_data["qcel_molecules"].append(mol)

        for key, value in alternative_reference.items():
            energies = []
            for j in value:
                energies.extend(j)
            df_data[key] = energies
        df = pd.DataFrame(df_data)
        df = update_colunn_names_to_match_bfdb(df)
        dfs[dataset_name] = df
        dfs[dataset_name].to_pickle(f"dfs/{dataset_name}.pkl")
    return dfs


def update_colunn_names_to_match_bfdb(df):
    if "corr_CCSD(T)/aDZ" not in df.columns and 'corr_CCSD(T)/haTZ' not in df.columns:
        print("No CCSD(T)/aDZ column found, skipping update.")
        return df
    mp2_col_names = [i for i in df.columns if ("MP2/" in i and "CBS" not in i)]
    ccsd_col_names = [i for i in df.columns if ("CCSD/" in i and "CBS" not in i)]
    ccsdt_col_names = [i for i in df.columns if ("CCSD(T)/" in i and "CBS" not in i)]
    for i in mp2_col_names:
        method = i.replace("corr_", "") + "/CP"
        hf_val = i.replace("corr_MP2", "HF")
        df[method] = df[i] + df[hf_val]
    for i in ccsd_col_names:
        method = i.replace("corr_", "") + "/CP"
        hf_val = i.replace("corr_CCSD", "HF")
        mp2_val = i.replace("corr_CCSD", "corr_MP2")
        df[method] = df[i] + df[hf_val] + df[mp2_val]
    for i in ccsdt_col_names:
        method = i.replace("corr_", "") + "/CP"
        hf_val = i.replace("corr_CCSD(T)", "HF")
        mp2_val = i.replace("corr_CCSD(T)", "corr_MP2")
        df[method] = df[i] + df[hf_val] + df[mp2_val]
    df["MP2/CBS/CP"] = df["HF/aQZ"] + df["corr_MP2/CBS(aTQZ)"]
    if 'corr_CCSD(T)/aDZ' in df.columns:
        df["CCSD(T)/CBS/CP"] = (
            df["HF/aQZ"]
            + df["corr_MP2/CBS(aTQZ)"]
            + df["corr_CCSD(T)/aDZ"]
            - df["corr_MP2/aDZ"]
        )
    else:
        df["CCSD(T)/CBS/CP"] = (
            df["HF/aQZ"]
            + df["corr_MP2/CBS(aTQZ)"]
            + df["corr_CCSD(T)/haTZ"]
            - df["corr_MP2/haTZ"]
        )
    print(df[['CCSD(T)/CBS/CP', 'MP2/CBS/CP']].describe())
    return df


def delta_model_errors(df, m1="HF/aug-cc-pVDZ/CP", m2="CCSD(T)/CBS/CP"):
    import apnet_pt
    from apnet_pt.pt_datasets.dapnet_ds import clean_str_for_filename

    m1_m2 = clean_str_for_filename(f"{m1}_{m2}")
    dapnet_dir = "/home/amwalla3/projects/AI4Science_QC/qcml_models/dap2"
    print(m1_m2)
    col = clean_str_for_filename(f"dAPNet2 {m1}")
    df[col] = apnet_pt.pretrained_models.dapnet2_model_predict(
        df["qcel_molecules"].tolist(),
        m1=m1,
        m2=m2,
        compile=False,
        use_GPU=False,
        pre_trained_model_path=f"{dapnet_dir}/{m1_m2}.pt",
    )
    return df, col


def drop_error_columns(df):
    error_vals = [
        "ELEMENT NOT SUPPORTED",
        "FRAGMENTATION ERROR",
        "MOL CONSTRUCTION ERROR",
    ]
    df = df[~df["processing_error"].isin(error_vals)].copy()
    return df


def update_column_names(df):
    df.rename(columns=lots, inplace=True)
    return df

def plot_violin_errors_per_df(df, output_filename="db_out.png"):
    from cdsg_plot import error_statistics

    df_labels_and_columns = {
        "HF/aug-cc-pVDZ/CP Error": "HF/aug-cc-pVDZ/CP Error",
        "dAPNet2+HF/aug-cc-pVDZ/CP Error": "dAPNet2+HF/aug-cc-pVDZ/CP Error",
        "HF/aug-cc-pVTZ/CP Error": "HF/aug-cc-pVTZ/CP Error",
        "dAPNet2+HF/aug-cc-pVTZ/CP Error": "dAPNet2+HF/aug-cc-pVTZ/CP Error",
        "HF/aug-cc-pVQZ/CP Error": "HF/aug-cc-pVQZ/CP Error",
        "dAPNet2+HF/aug-cc-pVQZ/CP Error": "dAPNet2+HF/aug-cc-pVQZ/CP Error",
        "MP2/aug-cc-pVDZ/CP Error": "MP2/aug-cc-pVDZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVDZ/CP Error": "dAPNet2+MP2/aug-cc-pVDZ/CP Error",
        "MP2/aug-cc-pVTZ/CP Error": "MP2/aug-cc-pVTZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVTZ/CP Error": "dAPNet2+MP2/aug-cc-pVTZ/CP Error",
        "MP2/aug-cc-pVQZ/CP Error": "MP2/aug-cc-pVQZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVQZ/CP Error": "dAPNet2+MP2/aug-cc-pVQZ/CP Error",
    }
    dfs = [
        {
            "df": df,
            "label": "",
            "ylim": [[-10.0, 10.0]],
        }
    ]
    print(df[['HF/aug-cc-pVDZ/CP Error', 'dAPNet2+HF/aug-cc-pVDZ/CP Error']].describe())
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total=df_labels_and_columns,
        output_filename=output_filename,
        figure_size=(8, 4),
        ylabel=r"IE Error vs. CCSD(T)/CBS (kcal/mol)",
        grid_widths=[1],
        grid_heights=[0.2, 1],
    )
    return

def plot_violin_errors():
    from cdsg_plot import error_statistics

    df_labels_and_columns = {
        "HF/aug-cc-pVDZ/CP Error": "HF/aug-cc-pVDZ/CP Error",
        "dAPNet2+HF/aug-cc-pVDZ/CP Error": "dAPNet2+HF/aug-cc-pVDZ/CP Error",
        "HF/aug-cc-pVTZ/CP Error": "HF/aug-cc-pVTZ/CP Error",
        "dAPNet2+HF/aug-cc-pVTZ/CP Error": "dAPNet2+HF/aug-cc-pVTZ/CP Error",
        "HF/aug-cc-pVQZ/CP Error": "HF/aug-cc-pVQZ/CP Error",
        "dAPNet2+HF/aug-cc-pVQZ/CP Error": "dAPNet2+HF/aug-cc-pVQZ/CP Error",
        "MP2/aug-cc-pVDZ/CP Error": "MP2/aug-cc-pVDZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVDZ/CP Error": "dAPNet2+MP2/aug-cc-pVDZ/CP Error",
        "MP2/aug-cc-pVTZ/CP Error": "MP2/aug-cc-pVTZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVTZ/CP Error": "dAPNet2+MP2/aug-cc-pVTZ/CP Error",
        "MP2/aug-cc-pVQZ/CP Error": "MP2/aug-cc-pVQZ/CP Error",
        "dAPNet2+MP2/aug-cc-pVQZ/CP Error": "dAPNet2+MP2/aug-cc-pVQZ/CP Error",
    }
    master_df = []
    for f in glob.glob("dfs/*_dapnet.pkl"):
        df = pd.read_pickle(f)
        master_df.append(df)
        dfs = [
            {
                "df": df,
                "label": "",
                "ylim": [[-2.0, 10.0] for i in range(5)],
            }
        ]
        print(f)
        print(df[['HF/aug-cc-pVDZ/CP Error', 'dAPNet2+HF/aug-cc-pVDZ/CP Error']].describe())
        print(len(df), df[['processing_error']])
        db_name = f.split("/")[-1].replace("_dapnet.pkl", ".png")
        error_statistics.violin_plot_table_multi_SAPT_components(
            dfs,
            df_labels_and_columns_total=df_labels_and_columns,
            output_filename=db_name,
            figure_size=(8, 4),
            ylabel=r"IE Error vs. CCSD(T)/CBS (kcal/mol)",
            grid_widths=[1],
            grid_heights=[0.2, 1],
        )

    dfs = [
        {
            "df": pd.concat(master_df, ignore_index=True),
            "label": "",
            "ylim": [[-2.0, 10.0] for i in range(5)],
        }
    ]
    db_name = f.split("/")[-1].replace("_dapnet.pkl", ".png")
    error_statistics.violin_plot_table_multi_SAPT_components(
        dfs,
        df_labels_and_columns_total=df_labels_and_columns,
        output_filename="all_dbs.png",
        figure_size=(8, 4),
        ylabel=r"IE Error vs. CCSD(T)/CBS (kcal/mol)",
        grid_widths=[1],
        grid_heights=[0.2, 1],
    )
    return


def dapnet_errors():
    # for f in glob.glob("dfs/*.pkl"):
    for f in glob.glob("dfs/NCIA_R739x5.pkl"):
        if "_dapnet" in f:
            continue
        print(f)
        df = pd.read_pickle(f)
        df = update_colunn_names_to_match_bfdb(df)
        df = update_column_names(df)
        pp(df.columns.values.tolist())
        len_df = len(df)
        df = drop_error_columns(df)
        print(f"Dropped {len_df - len(df)} rows with processing errors ({len_df})")
        for k in lots.values():
            print(k)
            try:
                df[f"{k} Error"] = df[k] - df["CCSD(T)/CBS/CP"]
                df, col = delta_model_errors(df, m1=k, m2="CCSD(T)/CBS/CP")
                df[f"dAPNet2+{k} Error"] = df[f"{k} Error"] + df[col]
                print(df[[f"{k} Error", f"dAPNet2+{k} Error"]].describe())
            except KeyError as e:
                print("KeyError: ", k, e)
                df[f"{k} Error"] = [0.0 for i in range(len(df))]
                df[f"dAPNet2+{k} Error"] = [0.0 for i in range(len(df))]
        db_name = f.split("/")[-1].replace(".pkl", ".png")
        plot_violin_errors_per_df(df, db_name)
        df.to_pickle(f.replace(".pkl", "_dapnet.pkl"))
    return


def main():
    # dfs = create_pandas_dfs()
    # pp(dfs)
    # return
    dapnet_errors()
    plot_violin_errors()
    return


if __name__ == "__main__":
    main()
