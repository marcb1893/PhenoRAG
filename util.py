import json
import os
import re
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer

project_path = os.path.dirname(__file__) + "/"
wnl = WordNetLemmatizer()
hpo_json_path = project_path + "/resources/util/hpo.json"
stopwords_file_path = project_path + "/resources/util/stopwords.txt"
num2word_file_path = project_path + "/resources/util/NUM.txt"

"""
util.py re-used from previous work of Feng et al., 2023
@article{Feng2023,
  title = {PhenoBERT: A Combined Deep Learning Method for Automated Recognition of Human Phenotype Ontology},
  volume = {20},
  ISSN = {2374-0043},
  url = {http://dx.doi.org/10.1109/TCBB.2022.3170301},
  DOI = {10.1109/tcbb.2022.3170301},
  number = {2},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Feng,  Yuhao and Qi,  Lei and Tian,  Weidong},
  year = {2023},
  month = mar,
  pages = {1269–1277}
}
"""


class HPO_class:
    """
    HPO节点类
    """

    def __init__(self, dic):
        # 解析dic
        self.id = dic["Id"]
        self.name = dic["Name"]
        self.alt_id = dic["Alt_id"]
        self.definition = dic["Def"]
        self.comment = dic["Comment"]
        self.synonym = dic["Synonym"]
        self.xref = dic["Xref"]
        self.is_a = dic["Is_a"]
        self.father = set(dic["Father"].keys())  # 祖先
        self.child = set(dic["Child"].keys())  # 子孙
        self.son = set(dic["Son"].keys())


class HPOTree:
    """
    构造HPO有向无环图结构的类；默认以HP:0000118为根节点
    """

    def __init__(self):

        with open(hpo_json_path, encoding="utf-8") as json_file:
            self.data = json.loads(json_file.read())

        # # 统计phenotypic abnomality下的分布
        # tmp_list = [[HPO_class(self.data[i]).name, len(HPO_class(self.data[i]).child)] for i in
        #             HPO_class(self.data["HP:0000118"]).son]
        # tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)
        # print(tmp_list)

        # print(HPO_class(self.data["HP:0000001"]).father)
        # print("HP:0000001" in HPO_class(self.data["HP:0000001"]).child)

        self.root = "HP:0000118"
        # self.phenotypic_abnormality = HPO_class(self.data[self.root]).child
        self.phenotypic_abnormality = HPO_class(self.data[self.root]).child
        # 没有root的HPO表型异常节点集合
        self.phenotypic_abnormalityNT = set(list(self.phenotypic_abnormality))
        self.phenotypic_abnormality.add(self.root)
        # get MIN node depth in HP:0000118(0) （due to a concept may has multi-inheritance）
        self.hpo_list = sorted(list(self.phenotypic_abnormality))
        self.n_concept = len(self.hpo_list)
        self.hpo2idx = {hpo: idx for idx, hpo in enumerate(self.hpo_list)}
        # Add None
        self.hpo2idx["None"] = len(self.hpo_list)
        self.idx2hpo = {self.hpo2idx[hpo]: hpo for hpo in self.hpo2idx}
        self.alt_id_dict = {}
        self.p_phrase2HPO = {}  # 用于短语的直接对应
        self.depth = 0  # 根节点深度为0
        self.layer1 = sorted(list(HPO_class(self.data[self.root]).son))  # 所有的layer1
        self.layer1_set = set(self.layer1)
        self.n_concept_l1 = len(self.layer1)
        self.hpo2idx_l1 = {hpo: idx for idx, hpo in enumerate(self.layer1)}
        # Add None
        self.hpo2idx_l1["None"] = len(self.layer1)
        self.idx2hpo_l1 = {self.hpo2idx_l1[hpo]: hpo for hpo in self.hpo2idx_l1}

        for hpo_name in self.data:
            struct = HPO_class(self.data[hpo_name])
            alt_ids = struct.alt_id
            for sub_alt_id in alt_ids:
                self.alt_id_dict[sub_alt_id] = hpo_name
            phrases = getNames(struct)
            for phrase in phrases:
                key = " ".join(sorted(processStr(phrase)))
                self.p_phrase2HPO[key] = hpo_name

    def buildHPOTree(self):
        """
        用BFS构建深度哈希对应表
        """

        self.depth_dict = {}
        queue = {self.root}
        visited = {self.root}
        depth = 0
        while len(queue) > 0:
            tmp = set()
            for node in queue:
                self.depth_dict[node] = depth
                son = HPO_class(self.data[node]).son
                for sub_node in son:
                    if sub_node not in visited:
                        visited.add(sub_node)
                        tmp.add(sub_node)
            queue = tmp
            depth += 1
        self.depth = depth - 1

    def getNameByHPO(self, hpo_num):
        """
        根据HPO号获得Concept Name
        :param hpo_num:
        :return:
        """
        return HPO_class(self.data[hpo_num]).name[0].lower()

    def getFatherHPOByHPO(self, hpo_num):
        """
        根据HPO号获得Concept的直属父节点的HPO号
        :param hpo_num:
        :return:
        """
        if hpo_num not in self.phenotypic_abnormalityNT:
            return None
        return HPO_class(self.data[hpo_num]).is_a

    def getLayer1HPOByHPO(self, hpo_num):
        """
        根据HPO号获得Concept的L1层祖先节点的HPO号
        :param hpo_num:
        :return:
        """
        if hpo_num not in self.phenotypic_abnormalityNT:
            return ["None"]
        if hpo_num in self.layer1_set:
            return [hpo_num]
        return list(self.layer1_set & HPO_class(self.data[hpo_num]).father)

    def getAllFatherHPOByHPO(self, hpo_num):
        """
        根据HPO号获得Concept的所有祖先节点的HPO号
        :param hpo_num:
        :return:
        """
        if hpo_num not in self.phenotypic_abnormalityNT:
            return set()
        return HPO_class(self.data[hpo_num]).father

    def getPhrasesByHPO(self, hpo_num):
        """
        根据HPO号获得Concept的Name & Synonyms
        :param hpo_num:
        :return:
        """
        return [i.lower() for i in getNames(HPO_class(self.data[hpo_num]))]

    def getAllPhrasesAbnorm(self):
        """
        获得所有表型异常Concept的Name & Synonyms
        :return:
        """
        phrases_list = []
        for hpo_name in self.hpo_list:
            phrases_list.extend(getNames(HPO_class(self.data[hpo_name])))
        return phrases_list

    def matchPhrase2HPO(self, phrase):
        """
        给定短语获得可能对应的HPO号；在注释用于直接的字典对应方式
        :param phrase:
        :return:
        """
        p_phrase = " ".join(sorted(processStr(phrase)))
        p_l_phrase = " ".join([WordItem.lemma_dict[i] if i in WordItem.lemma_dict else i for i in p_phrase.split()])
        if p_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_phrase]
        elif p_l_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_l_phrase]
        return ""

    def getHPO2idx(self, hpo_num):
        """
        返回hpo_num在表型异常根节点下对应的idx
        """
        return self.hpo2idx[hpo_num]

    def getHPO2idx_l1(self, hpo_num):
        """
        返回hpo_num在layer1节点中的对应的idx
        """
        return self.hpo2idx_l1[hpo_num]

    def getIdx2HPO(self, idx):
        """
        返回idx在在表型异常根节点下对应的hpo_num
        """
        return self.idx2hpo[idx]

    def getIdx2HPO_l1(self, idx):
        """
        返回idx在layer1节点中的对应的hpo_num
        """
        return self.idx2hpo_l1[idx]

    def getMaterial4L1(self, root_l1):
        """
        返回给定的L1的hpo_num，获得以该L1节点为根节点所构建DAG的各项对应表
        :param root_l1:
        :return:
        """
        root_idx = self.getHPO2idx_l1(root_l1)
        hpo_list = HPO_class(self.data[root_l1]).child
        hpo_list.add(root_l1)
        hpo_list = sorted(hpo_list)
        n_concept = len(hpo_list)
        hpo2idx = {hpo: idx for idx, hpo in enumerate(hpo_list)}
        # Add None
        hpo2idx["None"] = len(hpo_list)
        idx2hpo = {hpo2idx[hpo]: hpo for hpo in hpo2idx}
        return root_idx, hpo_list, n_concept, hpo2idx, idx2hpo

    def getNodeSimilarityByID(self, hpoNum1, hpoNum2):
        """
        计算hpoNum1和hpoNum2的节点相似性，基于edge和information content
        默认采用基于edge的方式
        """
        # Use phenotypic abnomality only; ensure score >= 0.0
        if hpoNum1 not in self.phenotypic_abnormality or hpoNum2 not in self.phenotypic_abnormality:
            return 0.0
        if hpoNum1 == self.root and hpoNum2 == self.root:
            return 1.0
        # depth in HPO Tree
        depth1 = self.depth_dict[hpoNum1]
        # print(depth1)    # HP:0000118 depth 0
        depth2 = self.depth_dict[hpoNum2]
        struct1 = HPO_class(self.data[hpoNum1])
        struct2 = HPO_class(self.data[hpoNum2])
        # get LCS
        father1 = struct1.father
        father1.add(hpoNum1)
        father2 = struct2.father
        father2.add(hpoNum2)
        ancestor = father1 & father2
        LCS = sorted(
            [[a, self.depth_dict[a]] for a in ancestor if a in self.phenotypic_abnormality],
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        depth3 = self.depth_dict[LCS]
        struct3 = HPO_class(self.data[LCS])
        # Edge-based score
        eb_score = 2 * depth3 / (depth1 + depth2)
        return eb_score

        # # Info-based score
        # pc1=(len(struct1.child)+1)/len(self.phenotypic_abnormality)
        # pc2=(len(struct2.child)+1)/len(self.phenotypic_abnormality)
        # pc3=(len(struct3.child)+1)/len(self.phenotypic_abnormality)
        # ib_score = 2*math.log2(pc3)/(math.log2(pc1)+math.log2(pc2))
        # return ib_score

    def getHPO_set_similarity_max(self, hpo_set1, hpo_set2):
        """
        计算HPO集合之间的相似性；使用最大值计算方式
        :param hpo_set1:
        :param hpo_set2:
        :return:
        """
        if len(hpo_set1) == 0 and len(hpo_set2) == 0:
            return 1.0
        if len(hpo_set1) == 0 or len(hpo_set2) == 0:
            return 0.0
        part1 = 0.0
        for hpo_num1 in hpo_set1:
            if hpo_num1 in hpo_set2:
                continue
            tmp = 0
            for hpo_num2 in hpo_set2:
                s_score = self.getNodeSimilarityByID(hpo_num1, hpo_num2)
                if s_score > tmp:
                    tmp = s_score
            part1 += 1 - tmp

        part2 = 0.0
        for hpo_num2 in hpo_set2:
            if hpo_num2 in hpo_set1:
                continue
            tmp = 0
            for hpo_num1 in hpo_set1:
                s_score = self.getNodeSimilarityByID(hpo_num1, hpo_num2)
                if s_score > tmp:
                    tmp = s_score
            part2 += 1 - tmp

        return 1 - ((part1 + part2) / len(hpo_set1 | hpo_set2))

    def getAdjacentMatrixAncestors(self, root_l1, num_nodes):
        """
        产生考虑所有祖先节点的邻接矩阵
        self.getAdjacentMatrixAncestorsAssist为辅助函数
        """
        import scipy.sparse as ss

        root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = self.getMaterial4L1(root_l1)
        ancestors_weight = {}
        for hpo_num in hpo_list:
            concept_id = hpo2idx[hpo_num]
            self.getAdjacentMatrixAncestorsAssist(ancestors_weight, concept_id, hpo2idx, idx2hpo)
        sparse_indexes = []
        sparse_values = []
        for concept_id in ancestors_weight:
            sparse_indexes.extend([[concept_id, ancestor_id] for ancestor_id in ancestors_weight[concept_id]])
            sparse_values.extend(
                [ancestors_weight[concept_id][ancestor_id] for ancestor_id in ancestors_weight[concept_id]]
            )

        indices = np.array(sparse_indexes)
        A = ss.coo_matrix(
            (np.array(sparse_values), (indices[:, 0], indices[:, 1])), shape=(num_nodes, num_nodes), dtype=np.float
        )
        return A.tocoo()

    def getAdjacentMatrixAncestorsAssist(self, ancestors_weight, concept_id, hpo2idx, idx2hpo):
        if concept_id in ancestors_weight:
            return ancestors_weight[concept_id].keys()
        ancestors_weight[concept_id] = {concept_id: 1.0}
        fathers = [i for i in HPO_class(self.data[idx2hpo[concept_id]]).is_a if i in hpo2idx]
        for father_hpo_num in fathers:
            father_id = hpo2idx[father_hpo_num]
            ancestors = self.getAdjacentMatrixAncestorsAssist(ancestors_weight, father_id, hpo2idx, idx2hpo)
            for ancestor_id in ancestors:
                if ancestor_id not in ancestors_weight[concept_id]:
                    ancestors_weight[concept_id][ancestor_id] = 0.0
                ancestors_weight[concept_id][ancestor_id] += ancestors_weight[father_id][ancestor_id] / len(fathers)
        return ancestors_weight[concept_id].keys()


def getNames(struct):
    """
    返回某一hpo结构中的name+synonym，并简单处理
    """
    # 训练集中同义词部分
    names = struct.name
    synonyms = struct.synonym
    # all name + synonym
    names.extend(synonyms)
    # 去重
    names = list(set(names))
    return names


def processStr(string):
    """
    输入字符串，返回经过符号处理的小写的word list
    :param string:
    :return:
    """
    string = re.sub("(?<=[A-Z])-(?=[\d])", "", string)  # 统一类型表述
    string = strip_accents(string.lower())
    string = re.sub("[-_\"'\\\\\t\r\n‘’]", " ", string)
    all_text = string.strip().split()
    return all_text


def strip_accents(s):
    """
    去除口音化，字面意思
    :param s:
    :return:
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


class WordItem:
    """
    英文单词的包装类
    """

    # 用于lemmatize
    lemma_dict = {}

    def __init__(self, text, start, end):
        self.text = text.lower()
        self.start = start
        self.end = end


def getNum2Word(file_path):
    Num2Word = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            inter = line.strip().split("\t")
            Num2Word[inter[0]] = inter[1]
    return Num2Word


class PhraseItem:
    """
    英文短语的包装类，包含了文本信息和起止点信息
    """

    Num2Word = getNum2Word(num2word_file_path)
    StopWords = stopwords.words("english")

    def __init__(self, word_items):
        self.word_items = word_items
        self.simple_items = []
        self.simplify()
        self.locs_set = set([i.start for i in word_items])
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False

    def simplify(self):
        """
        对phrase_item进行简化，去除常用词以及替换数字
        :return:
        """
        for word_item in self.word_items:
            if word_item.text in PhraseItem.Num2Word:
                self.simple_items.append(WordItem(PhraseItem.Num2Word[word_item.text], word_item.start, word_item.end))
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                self.simple_items.append(word_item)

    def toString(self):
        return " ".join([i.text for i in self.word_items])

    def toSimpleString(self):
        return " ".join([i.text for i in self.simple_items])

    def include(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(phrase_item.locs_set):
            return True
        return False

    def issubset(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set):
            return True
        return False

    def set_no_flag(self):
        self.no_flag = True

    def __len__(self):
        return len(self.word_items)


def getNames(struct):
    """
    返回某一hpo结构中的name+synonym，并简单处理
    """
    # 训练集中同义词部分
    names = struct.name
    synonyms = struct.synonym
    # all name + synonym
    names.extend(synonyms)
    # 去重
    names = list(set(names))
    return names


def strip_accents(s):
    """
    去除口音化，字面意思
    :param s:
    :return:
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def processStr(string):
    """
    输入字符串，返回经过符号处理的小写的word list
    :param string:
    :return:
    """
    string = re.sub("(?<=[A-Z])-(?=[\d])", "", string)  # 统一类型表述
    string = strip_accents(string.lower())
    string = re.sub("[-_\"'\\\\\t\r\n‘’]", " ", string)
    all_text = string.strip().split()
    return all_text


def isNum(strings):
    """
    判断给定字符串是否为数字
    :param strings:
    :return:
    """
    try:
        float(strings)
        return True
    except ValueError:
        return False


def containNum(strings):
    """
    判断给定字符串是否包含数字
    :param strings:
    :return:
    """
    for c in strings:
        if c.isdigit():
            return True
    return False


def getStopWords():
    """
    返回stopwords_file_path给定的stop words
    :return:
    """
    stopwords = set()
    with open(stopwords_file_path, encoding="utf-8") as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords


def getSpliters():
    """
    用于分割短句的分割词
    :return:
    """
    spliters = set(
        [word for word, pos in nltk.pos_tag(stopwords.words("english")) if pos in {"CC", "WP", "TO", "WDT"}]
        + [",", ".", ":", ";", "(", ")", "[", "]", "/"]
    )
    return spliters


def getNegativeWords():
    negatives = {"no", "not", "none", "negative", "non", "never", "few", "lower", "fewer", "less", "barely", "normal"}
    return negatives


class SpanTokenizer:
    """
    基于NLTK工具包，自定义一个详细版本的Tokenizer
    """

    def __init__(self):
        self.tokenizer_big = PunktSentenceTokenizer()
        self.tokenizer_small = TreebankWordTokenizer()

    def tokenize(self, text):
        result = []
        sentences_span = self.tokenizer_big.span_tokenize(text)
        for start, end in sentences_span:
            sentence = text[start:end]
            tokens_span = self.tokenizer_small.span_tokenize(sentence)
            for token_start, token_end in tokens_span:
                result.append([start + token_start, start + token_end])
        return result


def process_text2phrases(text, clinical_ner_model):
    """
    用于从文本中提取Clinical Text Segments
    :param text:自由文本
    :param clinical_ner_model: Stanza提供的预训练NER模型
    :return: List[PhraseItem]
    """
    tokenizer = SpanTokenizer()
    spliters = getSpliters()
    stopwords = getStopWords()
    # 将文本处理成正常的小写形式
    text = strip_accents(text.lower())
    text = re.sub("[-_\"'\\\\\t‘’]", " ", text)
    # 对于换行符替换为句号，后续作为分割词
    text = re.sub("(?<=[\w])[\r\n]", ".", text)

    clinical_docs = clinical_ner_model(text)
    sub_sentences = []

    for sent_c in clinical_docs.sentences:
        clinical_tokens = sent_c.tokens

        # Stanza
        flag = False
        curSentence = []
        tmp = set()
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            if "PROBLEM" in clinical_tokens[i].ner and wi.text not in {",", ".", ":", ";", "(", ")", "[", "]"}:
                curSentence.append(wi)
            else:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    sub_sentences.append(phrase_item)
                    tmp.update(phrase_item.locs_set)
                    flag = True
                curSentence = []

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            sub_sentences.append(phrase_item)
            tmp.update(phrase_item.locs_set)
            flag = True

        # phrase segmentation
        # 只有Stanza标记的句子才作补充
        if not flag:
            continue
        curSentence = []
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            # 用于后续lemma比对
            text_lemma = wnl.lemmatize(clinical_tokens[i].text)
            if clinical_tokens[i].text not in WordItem.lemma_dict:
                WordItem.lemma_dict[clinical_tokens[i].text] = text_lemma
            if clinical_tokens[i].text in spliters:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    # 只有不与Stanza重叠的部分才加入
                    if len(phrase_item.locs_set & tmp) == 0:
                        sub_sentences.append(phrase_item)
                curSentence = []
            else:
                curSentence.append(wi)

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            if len(phrase_item.locs_set & tmp) == 0:
                sub_sentences.append(phrase_item)

    # 否定检测
    for phrase_item in sub_sentences:
        flag = False
        for token in phrase_item.word_items:
            if token.text.lower() in {
                "no",
                "not",
                "none",
                "negative",
                "non",
                "never",
                "few",
                "lower",
                "fewer",
                "less",
                "normal",
            }:
                flag = True
                break
        if flag:
            phrase_item.set_no_flag()

    # 省略恢复
    sub_sentences_ = []
    for idx, pi in enumerate(sub_sentences):
        # 将含有and, or, / 的短句用tokenize进行拆分
        sub_locs = [
            [i + pi.start_loc, j + pi.start_loc] for i, j in tokenizer.tokenize(text[pi.start_loc : pi.end_loc])
        ]
        sub_phrases = []
        curr_phrase = []
        # 把以and, or, / 分割的短语提出
        for loc in sub_locs:
            wi = WordItem(text[loc[0] : loc[1]], loc[0], loc[1])
            if wi.text in {"and", "or", "/"}:
                if len(curr_phrase) > 0:
                    sub_phrases.append(PhraseItem(curr_phrase))
                    sub_phrases[-1].no_flag = pi.no_flag
                curr_phrase = []
            else:
                curr_phrase.append(wi)
        if len(curr_phrase) > 0:
            sub_phrases.append(PhraseItem(curr_phrase))
            sub_phrases[-1].no_flag = pi.no_flag

        # 首先把原始分割的短语都加进去
        for item in sub_phrases:
            sub_sentences_.append(item)

        # 只考虑A+B形式的恢复
        if len(sub_phrases) == 2:
            if len(sub_phrases[0]) >= 1 and len(sub_phrases[1]) == 1:
                tmp = sub_phrases[0].word_items[:-1][:]
                tmp.extend(sub_phrases[1].word_items)
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag
            elif len(sub_phrases[0]) == 1 and len(sub_phrases[1]) >= 1:
                tmp = sub_phrases[0].word_items[:]
                tmp.extend(sub_phrases[1].word_items[1:])
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag

    sub_sentences = sub_sentences_

    # print([i.toSimpleString() for i in sub_sentences])

    # 穷举短语 删除纯数字
    phrases_list = []
    for pi in sub_sentences:
        tmp = pi.toSimpleString()
        if isNum(tmp) or len(tmp) <= 1:
            continue
        for i in range(len(pi.simple_items)):
            for j in range(10):
                if i + j == len(pi.simple_items):
                    break
                if len(pi.simple_items[i : i + j + 1]) == 1:
                    tmp_str = pi.simple_items[i : i + j + 1][0].text
                    if tmp_str in stopwords or isNum(tmp_str):
                        continue
                phrases_list.append(PhraseItem(pi.simple_items[i : i + j + 1]))
                phrases_list[-1].no_flag = pi.no_flag

    # print(len(phrases_list))
    # print([i.toString() for i in phrases_list])
    return phrases_list
