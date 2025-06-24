
import pickle
import pandas as pd
from utils.standardization import *
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.QED import qed
from utils.sascore import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')
REMOVE_HS = lambda x: Chem.RemoveHs(x, sanitize=False)


def log_error(err):
    print(err)
    return None


def conformer_match(smiles, new_dihedrals):
    '''convert it like confs'''
    mol_rdkit = Chem.MolFromSmiles(smiles)

    AllChem.EmbedMultipleConfs(mol_rdkit, numConfs=1)
    if mol_rdkit.GetNumConformers() ==0:
        print('wrong:',smiles)

    rotable_bonds = get_torsion_angles(mol_rdkit)

    if not rotable_bonds:
        return log_error("no_rotable_bonds")

    new_rdkit = apply_changes(mol_rdkit, new_dihedrals[:len(rotable_bonds)], rotable_bonds, 0)

    return new_rdkit


def read_data(path):
    data = []
    with open(path, 'rb') as f:
        while True:
            try:
                aa = pickle.load(f)
                data.extend(aa)
            except EOFError:
                break
    return data


def get_mol(smiles_list):
    """将 SMILES 或 RDKit Mol 对象列表转换为 RDKit Mol 对象列表"""
    mols = []

    for item in smiles_list:
        if isinstance(item, Chem.Mol):  # 直接是 RDKit Mol 对象
            mols.append(item)
        else:  # 处理 SMILES 字符串
            try:
                mol = Chem.MolFromSmiles(item)  # 尝试解析 SMILES
                if mol:
                    Chem.SanitizeMol(mol)  # 规范化分子
                mols.append(mol)  # 无论 mol 是否成功解析，都 append
            except:  # 解析或规范化失败，存 None
                mols.append(None)

    return mols


def canonic_smiles(smiles_list):
    """将 SMILES 列表转换为规范的 SMILES 字符串列表"""
    mols = get_mol(smiles_list)
    return [Chem.MolToSmiles(mol) if mol else None for mol in mols]


def fraction_valid(smiles_list):
    """计算 SMILES 列表中有效分子的比例"""
    mols = get_mol(smiles_list)
    return 1 if not mols else 1 - mols.count(None) / len(mols)


def remove_invalid(smiles_list):
    """获取规范化的 SMILES 并过滤掉无效分子"""
    return [smi for smi in canonic_smiles(smiles_list) if smi]


def fraction_unique(smiles_list):
    """计算有效 SMILES 中的唯一 SMILES 的比例"""
    smiles_ = remove_invalid(smiles_list)
    unique_smiles = set(smiles_)
    return len(unique_smiles) / len(smiles_) if smiles_ else 0


def novelty(gen_list, train_list):
    """计算生成的分子在训练集之外的新颖性比例"""
    train_set = set(canonic_smiles(train_list))  # 训练数据规范化
    gen_smiles_set = {smi for smi in canonic_smiles(gen_list) if smi}  # 过滤 None
    return len(gen_smiles_set - train_set) / len(gen_smiles_set) if gen_smiles_set else 0


def bulk_similarity(smiles: list, target_smiles: list, target_fps: list = None, metric: str = 'tanimoto'):
    """
    Calculates the similarity between a list of SMILES and a target list of SMILES
    using Bulk Tanimoto or Dice similarity
    Arguments:
        smiles (list)             : list of SMILES
        target (list)             : list of SMILES
        metric (str)              : similarity metric (tanimoto or dice)
    """

    if metric == 'tanimoto':
        metric_func = DataStructs.TanimotoSimilarity
    elif metric == 'dice':
        metric_func = DataStructs.DiceSimilarity
    else:
        raise ValueError('Invalid metric. Choose "tanimoto" or "dice".')

    # Convert SMILES to RDKit molecules and compute fingerprints
    mols = [Chem.MolFromSmiles(smi) if smi != '---' else None for smi in smiles]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None for mol in mols]

    if target_fps is None:
        target_mols = [Chem.MolFromSmiles(smi) if smi != '---' else None for smi in target_smiles]
        target_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None for mol in target_mols]

    # Initialize similarity matrix
    similarity = np.zeros((len(smiles), len(target_smiles)))

    # Compute similarity
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        for j, target_fp in enumerate(target_fps):
            if target_fp is None:
                similarity[i, j] = 0
            else:
                try:
                    similarity[i, j] = metric_func(fp, target_fp)
                except Exception as e:
                    similarity[i, j] = 0

    return similarity


class Score:
    """
    Calculate the necessary properties for rewarding score.
    In this case, we apply a properties filter, QED score,
    stereochemisty clash and docking score with QuickVina2 as reward terms.
    You can also DIY any reward terms if you like :-)
    """
    # def __init__(self, mol, protein_dir, rank=None):
    def __init__(self, mol):
        self.mol = mol
        self.smi = Chem.MolToSmiles(mol)
        self.qed = qed(mol)
        self.sa = sascorer.calculateScore(mol)
        self.logp = MolLogP(mol)
        self.mw = Descriptors.MolWt(mol)
        #self.clash = check_geometry(mol)
        # self.dock = docking_score(mol, protein_dir, rank)
        # self.vina_score = self.dock['affinity']
        self.tpsa = CalcTPSA(mol)


def append_tar(gen_mol_file, tar_file, output_file, num):

    generate_mol = pd.read_csv(gen_mol_file, header=None)
    tar = pd.read_csv(tar_file, nrows=11)
    result_list = []

    for index, row in tar.iterrows():
        target_id1 = row['target_id1']
        target_id2 = row['target_id2']
        raw_smi = row['smiles']
        for _ in range(num):
            result_list.append([target_id1, target_id2, raw_smi])

    result_df = pd.DataFrame(result_list)
    final_df = pd.concat([generate_mol, result_df], axis=1, ignore_index=True)
    column_names = ['smiles', 'target_id1', 'target_id2', 'raw_smiles']
    final_df.to_csv(output_file, header=column_names, index=False)


# def calc_prop():

if __name__ == '__main__':

    csv_file = './generation/test_set1_gen.csv'
    column_names = ['smiles', 'target1', 'target2', 'smiles1', 'smiles2']
    generate_mol = pd.read_csv(csv_file, header=None, names=column_names)
    print(generate_mol.head())

    train_ = pd.read_csv('.data/cluster_2targets.csv')
    train_ = train_.drop_duplicates(subset=['smiles'])
    train_smi = train_['smiles']

    gen_smiles = []
    only_smiles = []
    QED = []
    LOGP = []
    MW = []
    SA = []
    TPSA = []
    rows_to_drop = []
    tar1 = []
    tar2 = []

    final_data = []

    for i, gen_mol in enumerate(generate_mol['smiles']):
        try:
            smiles, torsion = gen_mol.split('GEO')
            smiles = smiles.replace(' ', '')
            torsion = np.array(torsion.split(' ')[1:]).astype(np.float64)
            pred_mol = conformer_match(smiles, torsion)
            if pred_mol:
                smi = Chem.MolToSmiles(pred_mol)
                score = Score(pred_mol)

                final_data.append({
                    'smiles': gen_mol,
                    'target1': generate_mol['target1'][i],
                    'target2': generate_mol['target2'][i],
                    'smiles1': generate_mol['smiles1'][i],
                    'smiles2': generate_mol['smiles2'][i],
                    'gen_smiles': smi,
                    'QED': score.qed,
                    'LogP': score.logp,
                    'SA': score.sa,
                    'TPSA': score.tpsa,
                    'Weight': score.mw
                })
        except Exception as e:
            continue

    final_df = pd.DataFrame(final_data)
    final_df.to_csv('./generation/test_set1_prop.csv', header=True, index=False)

