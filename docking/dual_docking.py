import os.path as osp
import os
from pathlib import Path
from easydict import EasyDict
import subprocess
import shutil
import random
from utils.standard import *
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import traceback
RDLogger.DisableLog('rdApp.*')


def write_sdf(mol, sdf_path):

    # mol = Chem.MolFromSmiles(smiles)  # 1. 转为 Mol 对象
    # if mol is None:
    #     raise ValueError(f"无法从 SMILES 构建分子：{smiles}")

    writer = Chem.SDWriter(sdf_path)  # 2. 初始化 SDF writer
    writer.write(mol)                 # 3. 写入 Mol
    writer.close()


def prepare_target(work_dir, protein_file_name, verbose=0):
    '''
    work_dir is the dir which .pdb locates
    protein_file_name: .pdb file which contains the protein data
    /home/qunsu/software/ADFRsuite_x86_64Linux_1.0/myFolder/bin/prepare_receptor
    '''
    protein_file = osp.join(work_dir, protein_file_name)

    command = '/home/gouqiaolin/ADFRsuite_x86_64Linux_1.0/myFolder/bin/prepare_receptor -r {protein} -o {protein_pdbqt}'.format(
        protein=protein_file,
        protein_pdbqt=protein_file + 'qt')
    if osp.exists(protein_file + 'qt'):
        return protein_file + 'qt'

    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()

    if verbose:
        if osp.exists(protein_file + 'qt'):
            print('successfully prepare the target')
        else:
            print('failed')

    return protein_file + 'qt'


def prepare_ligand(work_dir, lig_sdf, verbose=1):

    lig_name = lig_sdf
    lig_mol2 = lig_sdf[:-3] + 'mol2'
    lig_mol2 = os.path.basename(lig_mol2)
    now_cwd = os.getcwd()
    lig_sdf = osp.join(work_dir, lig_sdf)
    cwd_mol2 = osp.join(now_cwd, lig_mol2)
    work_mol2 = osp.join(work_dir, lig_mol2)

    command = f'''/home/gouqiaolin/miniconda3/envs/my3/bin/obabel {lig_sdf} -O {work_mol2}'''

    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=work_dir
    )
    stdout, stderr = proc.communicate()

    os.remove(lig_sdf)

    shutil.copy(work_mol2, now_cwd)

    lig_pdbqt = lig_name[:-3] + 'pdbqt'
    lig_pdbqt = os.path.basename(lig_pdbqt)
    cwd_pdbqt = osp.join(now_cwd, lig_pdbqt)
    work_pdbqt = osp.join(work_dir, lig_pdbqt)

    command = f'/home/gouqiaolin/ADFRsuite_x86_64Linux_1.0/myFolder/bin/prepare_ligand -l {work_mol2} - A hydrogens - o {cwd_pdbqt}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    os.remove(cwd_mol2)
    os.remove(work_mol2)
    if osp.exists(work_pdbqt):
        os.remove(work_pdbqt)
    shutil.move(cwd_pdbqt, work_dir)
    if os.path.exists(lig_pdbqt):
        if verbose:
            print('prepare successfully !')
        else:
            print('generation failed!')
    return lig_pdbqt


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:, 0].mean()
    centroid_y = lig_xyz[:, 1].mean()
    centroid_z = lig_xyz[:, 2].mean()
    return centroid_x, centroid_y, centroid_z


def docking_with_sdf(work_dir, protein_pdbqt, lig_pdbqt, centroid, verbose=0, out_lig_sdf=None, save_pdbqt=True):
    '''

    /home/gouqiaolin/miniconda3/envs/my3/bin/obabel

    work_dir: is same as the prepare_target
    protein_pdbqt: .pdbqt file
    lig_sdf: ligand .sdf format file
    '''

    lig_pdbqt = osp.join(work_dir, lig_pdbqt)
    protein_pdbqt = osp.join(work_dir, protein_pdbqt)
    cx, cy, cz = centroid
    out_lig_sdf_dirname = osp.dirname(lig_pdbqt)
    out_lig_pdbqt_filename = osp.basename(lig_pdbqt).split('.')[0] + '_out.pdbqt'
    out_lig_pdbqt = osp.join(out_lig_sdf_dirname, out_lig_pdbqt_filename)
    if out_lig_sdf is None:
        out_lig_sdf_filename = osp.basename(lig_pdbqt).split('.')[0] + '_out.sdf'
        out_lig_sdf = osp.join(out_lig_sdf_dirname, out_lig_sdf_filename)
    else:
        out_lig_sdf = osp.join(work_dir, out_lig_sdf)

    command = '''qvina2 \
        --receptor {receptor_pre} \
        --ligand {ligand_pre} \
        --center_x {centroid_x:.4f} \
        --center_y {centroid_y:.4f} \
        --center_z {centroid_z:.4f} \
        --size_x 20 --size_y 20 --size_z 20 \
        --cpu 40 \
        --out {out_lig_pdbqt} \
        --exhaustiveness {exhaust}
        /home/gouqiaolin/miniconda3/envs/my3/bin/obabel {out_lig_pdbqt} -O {out_lig_sdf} -h'''.format(
        receptor_pre=protein_pdbqt,
        ligand_pre=lig_pdbqt,
        centroid_x=cx,
        centroid_y=cy,
        centroid_z=cz,
        out_lig_pdbqt=out_lig_pdbqt,
        exhaust=30,
        out_lig_sdf=out_lig_sdf)
    proc = subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    proc.communicate()

    os.remove(lig_pdbqt)
    if not save_pdbqt:
        os.remove(out_lig_pdbqt)

    print('out_lig_sdf:', out_lig_sdf)

    if verbose:
        if os.path.exists(out_lig_sdf):
            print('searchable docking is finished successfully')
        else:
            print('docing failed')
    return out_lig_sdf


def get_result(docked_sdf, ref_mol=None):

    suppl = Chem.SDMolSupplier(docked_sdf, sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({'affinity': float(line[0])}))
        if i == 0:
            break

    return results[0]


def dual_docking_score(protein_pdb1, ligand_sdf1, smiles1_ref, protein_dir, save_sdf=False) -> float:
    """
    Calculate the docking score for a given protein and ligand.

    Parameters:
    protein_pdb1: str - Path to the protein .pdb file.
    ligand_sdf1: str - Path to the ligand .sdf file.
    smiles1_ref: str - The SMILES of the ligand.
    protein_dir: str - Directory containing the protein and ligand files.
    save_sdf: bool - Whether to save the generated SDF file (default is False).

    Returns:
    float - The docking score.
    """
    # Prepare the protein
    try:
        protein_pdbqt = prepare_target(protein_dir, protein_pdb1)  # Prepare protein for docking

        # Construct sdf file for the ligand (using the given SMILES)
        random_id = str(random.randint(1000, 2000))
        sdf_file = f'gene_{random_id}.sdf'
        sdf_dir = os.path.join(protein_dir, sdf_file)
        write_sdf(smiles1_ref, sdf_dir)  # Write ligand SDF from SMILES

        # Prepare the ligand
        lig_pdbqt = prepare_ligand(protein_dir, sdf_file) # smiles1_ref

        # Get the centroid of docking (based on the original ligand SDF file)
        centroid = sdf2centroid(ligand_sdf1)

        # Perform docking
        docked_sdf = docking_with_sdf(str(protein_dir), protein_pdbqt, lig_pdbqt, centroid)

        # Get docking result (score)
        result = get_result(docked_sdf, ref_mol=smiles1_ref)  # Assuming `get_result` uses the SMILES as reference

        # Optionally save docking results in sdf format
        if not save_sdf:
            Path(docked_sdf).unlink()

        return result

    except:
         print("docking failed")
         traceback.print_exc()


def log_error(err):
    print(err)
    return None


def conformer_match(smiles, new_dihedrals):
    '''convert it like confs'''
    mol_rdkit = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=1)
    if mol_rdkit.GetNumConformers() == 0:
        print('wrong:', smiles)
        return log_error("no_conformer")

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        new_rdkit = mol_rdkit
        # return log_error("no_rotable_bonds")

    else:
        n_bonds = len(rotable_bonds)
        if len(new_dihedrals) < n_bonds:
            new_dihedrals += [0.0] * (n_bonds - len(new_dihedrals))
        elif len(new_dihedrals) > n_bonds:
            new_dihedrals = new_dihedrals[:n_bonds]

        new_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, 0)

    return new_rdkit


if __name__ == '__main__':

    csv_file_1 = '../generation/test_set1_gen.csv'
    df_1 = pd.read_csv(csv_file_1)
    print(df_1.head())

    csv_file_2 = './data/dock_ref/output.csv'
    df_2 = pd.read_csv(csv_file_2)
    df_2.columns = ['smiles1', 'smiles2', 'target1', 'target2']

    rename_csv = './data/dock_ref/rename.csv'
    df_rename = pd.read_csv(rename_csv)
    ligand_protein_folder = '/home/gouqiaolin/dataset/Dualdiff_data/dock/ligand_protein_dataset_v2/'

    df_1['target1_score'] = 0.0
    df_1['target2_score'] = 0.0

    for index, row in df_1.iterrows():
        gen_mol = row['smiles']

        try:
            smiles, torsion = gen_mol.split('GEO')
            smiles = smiles.replace(' ', '')
            torsion = np.array(torsion.split(' ')[1:]).astype(np.float64)
            smiles1 = conformer_match(smiles, torsion)

            row['target1'] = row['target1'].replace('_WT', '')
            row['target2'] = row['target2'].replace('_WT', '')

            target1 = row['target1']
            target2 = row['target2']

            res_smiles1 = row['smiles1']
            res_smiles2 = row['smiles2']

            matched_row = df_2[
                (df_2['target1'] == target1) &
                (df_2['target2'] == target2) &
                (df_2['smiles1'] == res_smiles1) &
                (df_2['smiles2'] == res_smiles2)
                ]

            print('matched_row:' ,len(matched_row))
            print(matched_row)

            if not matched_row.empty:

                smiles1_ref = matched_row.iloc[0]['smiles1']
                smiles2_ref = matched_row.iloc[0]['smiles2']

                subfolder1_index = df_rename[df_rename['Original Folder Name'] == smiles1_ref]['New Folder Name'].values
                subfolder2_index = df_rename[df_rename['Original Folder Name'] == smiles2_ref]['New Folder Name'].values

                if subfolder1_index.size > 0 and subfolder2_index.size > 0:
                    subfolder1 = os.path.join(ligand_protein_folder, str(subfolder1_index[0]))
                    subfolder2 = os.path.join(ligand_protein_folder, str(subfolder2_index[0]))

                    if os.path.isdir(subfolder1) and os.path.isdir(subfolder2):
                        subfolder1 = next(os.path.join(subfolder1, d) for d in os.listdir(subfolder1) if
                                          os.path.isdir(os.path.join(subfolder1, d)))
                        subfolder2 = next(os.path.join(subfolder2, d) for d in os.listdir(subfolder2) if
                                          os.path.isdir(os.path.join(subfolder2, d)))

                        protein_pdb1 = os.path.join(subfolder1, 'protein_clean.pdb')
                        ligand_sdf1 = os.path.join(subfolder1, 'ligand.sdf')

                        protein_pdb2 = os.path.join(subfolder2, 'protein_clean.pdb')
                        ligand_sdf2 = os.path.join(subfolder2, 'ligand.sdf')

                        if os.path.exists(protein_pdb1) and os.path.exists(ligand_sdf1) and os.path.exists(
                                protein_pdb2) and os.path.exists(ligand_sdf2):

                            score1 = dual_docking_score(protein_pdb1, ligand_sdf1, smiles1, subfolder1)
                            score2 = dual_docking_score(protein_pdb2, ligand_sdf2, smiles1, subfolder2)

                            if score1 and score2:
                                s1 = score1['affinity']
                                s2 = score2['affinity']
                                print(index, target1, target2, s1, s2)

                                df_1.at[index, 'target1_score'] = s1
                                df_1.at[index, 'target2_score'] = s2

                        else:
                            print(f"文件不存在：{protein_pdb1}, {ligand_sdf1}, {protein_pdb2}, {ligand_sdf2}")
                    else:
                        print(f"没有找到对应的子文件夹: {smiles1_ref} 或 {smiles2_ref}")
                else:
                    print(f"没有在rename.csv中找到匹配的子文件夹索引，目标: {smiles1_ref} 或 {smiles2_ref}")
            else:
                print(f"没有找到匹配的行，目标: {target1}, {target2}")

        except:
            print('invaild smiles')

    output_csv = 'test_set1_affi.csv'
    df_1.to_csv(output_csv, index=False)

    print(f"分数计算完成，已保存到 {output_csv}")


