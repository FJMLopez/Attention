import json
from dataclasses import dataclass
from typing import ClassVar, List

from config import HEAD_DOCUMENTS_FILE
from utils import get_documents_heads

@dataclass
class Sentence:
    """
    Représente une phrase avec ses tokens et son identifiant.

    Attributes:
        tokens (list[str]): Liste des tokens de la phrase.
        id (int): Identifiant de la phrase du modèle.
        heads (ClassVar[List[int]]): Liste des têtes de documents (chargée depuis un fichier).
    """

    tokens: List[str]
    system_id: int
    heads: ClassVar[List[int]] = get_documents_heads(HEAD_DOCUMENTS_FILE)

    ###########################
    ### Méthodes d'instance ###
    ###########################

    def __repr__(self) -> str:
        return f"{{'_system_id': {self.system_id}, '_tokens': {self.tokens}}}"
    
    #############################
    ### Propriétés d'instance ###
    #############################

    @property
    def system_data_id(self) -> int:
        """
        Retourne l'id du système pour cette phrase.
        
        Returns:
            int: ID du système.
        """
        return self.system_id

    @property
    def corpus_data_id(self) -> int:
        """
        Convertit l'ID système vers l'ID corpus, en ajoutant un décalage
        dépendant du numéro de document (déterminé via `heads`).
        
        Hypothèse : chaque entrée dans heads est le début d’un document,
        et le corpus ID = system_id + offset du document.

        Returns:
            int: ID du corpus.
        
        Example:
        >>> Sentence(tokens=["This", "is", "a", "test"], system_id=0).corpus_data_id
        1
        >>> Sentence(tokens=["Another", "sentence"], system_id=1).corpus_data_id
        2
        >>> Sentence(tokens=["Doc2", "starts", "here"], system_id=87).corpus_data_id
        88
        >>> Sentence(tokens=["Yet", "another", "one"], system_id=187).corpus_data_id
        187
        >>> Sentence(tokens=["More", "data"], system_id=337).corpus_data_id
        336
        >>> Sentence(tokens=["Final", "sentence"], system_id=356).corpus_data_id
        355
        >>> Sentence(tokens=["Last", "one"], system_id=491).corpus_data_id
        489
        >>> Sentence(tokens=["End", "of", "test"], system_id=1733).corpus_data_id
        1725
        """

        shift = -2
        i = 0
        if self.system_id == 0:
            return 1
        while self.system_id >= self.heads[i]:
            shift += 1
            i += 1
        return self.system_id - shift    
    
    #######################
    ### fonctionnalités ###
    #######################

    def len(self) -> int:
        return len(self.tokens)

    def append(self, value: str) -> None:
        """Ajoute un (ou plusieurs via liste) token(s) à la fin de la phrase

        Args:
            value (str or List[str]): token ou liste à ajouter à la fin de la phrase
        
        Tests:
        >>> s1 = Sentence(system_id= 3, tokens= ['Ce@@', 'ci', 'est', '<pad>', 'un'])
        >>> s1.append('test')
        >>> print(s1)
        {'_system_id': 3, '_tokens': ['Ce@@', 'ci', 'est', '<pad>', 'un', 'test']}
        >>> s1 = Sentence(system_id= 3, tokens= ['Ce@@', 'ci', 'est', 'test', '<pad>', 'un'])
        >>> s1.append(['test1', 'test2'])
        >>> print(s1)
        {'_system_id': 3, '_tokens': ['Ce@@', 'ci', 'est', 'test', '<pad>', 'un', 'test1', 'test2']}
        """
        assert isinstance(value, str) or (isinstance(value, list) and all([isinstance(val, str) for val in value])), f"[DEBUG] Snt().append only supports strings and List[string]. Current type : {type(value)}"
        if isinstance(value, str):
            self.tokens.append(value)
        elif isinstance(value, List):
            self.tokens += value

    def insert(self, index: int, value) -> None:
        """Insert value en position index dans la liste de tokens

        Args:
            index (int): position de l'index où inserer value
            value (str): token unique à insérer
        
        Tests:
        >>> s1 = Sentence(system_id= 3, tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> s1.insert(0, "test")
        >>> print(s1)
        {'_system_id': 3, '_tokens': ['test', 'Ce@@', 'ci', 'est', '<pad>', 'un', 'te@@', 'st', '.', '<eos>']}
        """
        assert isinstance(index, int), f"index must be an int. Current type: {type(index)}"
        assert isinstance(value, str), f"value must be a str. Current type: {type(value)}"
        self.tokens.insert(index, value)

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)

    ##########################
    ### Méthodes statiques ###
    ##########################
    @staticmethod
    def list_suppr_pad(tokens, padding_mark="<pad>", strict=False)-> List[int]:
        """retourne la liste des index du padding à supprimer par ordre décroissant.

        Args:
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        Example
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=False)
        [6, 2, 1]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=True)
        [6, 2, 1, 0]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="ci", strict=True)
        [4]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="padding", strict=True)
        []
        """
        stop = -1 if strict else 0
        list_suppr_pad = []
        for i in range(len(tokens)-1, stop, -1):
            if tokens[i] == padding_mark:
                list_suppr_pad.append(i)
        return list_suppr_pad

    @staticmethod
    def list_fusion_bpe(tokens: List[str], BPE_mark: str = '@@') -> List[List[int]]:
        """retourne la liste décroissante des tokens contenant une marque de BPE à la fin

        Returns:
            List[int]: liste décroissante des tokens contenant une marque de BPE à la fin
        >>> Sentence.list_fusion_bpe(tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        [[8], [7], [6, 5], [4], [3], [2], [1, 0]]
        >>> Sentence.list_fusion_bpe(tokens= ["lu@@", "bu@@", "lu@@", "le", ".", "<eos>"])
        [[5], [4], [3, 2, 1, 0]]
        >>> Sentence.list_fusion_bpe(tokens= ["lu@@", "bu", "lu@@", "le", ".", "<eos>"])
        [[5], [4], [3, 2], [1, 0]]
        

        # >>> Sentence.list_fusion_bpe(tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], BPE_mark="bpe_mark")
        """
        assert not tokens[-1].endswith(BPE_mark), f"Dernier token contenant une marque de BPE. Sentence: {tokens}"

        liste_bpe = []
        flag = False # Flag permettant de savoir si on est en train de traiter un mot BPEisé
        indice_token = 0
        while indice_token < len(tokens):
            # Pour chaque token de la phrase
            if flag == False: # Si le token précédant ne contient pas de marque de BPE (représenté par le flag)
                # On ajoute une nouvelle liste contenant l'indice du token courant (nouveau groupe)
                liste_bpe.append([indice_token])
            else: # On ajoute l'indice du token courant au groupe en cours
                liste_bpe[-1].append(indice_token)

            if tokens[indice_token].endswith(BPE_mark): # Si le token courant contient une marque de BPE
                # on ajoute une liste contenant l'indice du token 
                flag = True
            else: # Si le token courant ne contient pas de marque de BPE
                # On met le flag à False signifiant qu'on a fini de traiter un mot BPEisé
                flag = False

            indice_token += 1
        return [groupe[::-1] for groupe in liste_bpe[::-1]] if len(liste_bpe) >= 1 else []

    ##############################
    ### Méthodes de Traitement ###
    ##############################
    def suppr_pad(self, list_index: List[int] | None = None, padding_mark='<pad>', strict=True) -> List[int]:
        """Supprime le padding de la phrase et retourne une liste contenant les index supprimés dans l'ordre décroissant

        Args:
            list_index (List[int] | None): Liste des index à supprimer. Si None, on calcule la liste via la méthode statique list_suppr_pad. Defaults to None.
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to True.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        
        Examples:
        >>> s1 = Sentence(system_id= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> list_index = Sentence.list_suppr_pad(s1.tokens, padding_mark="<pad>", strict=False)
        >>> s1.suppr_pad(list_index)
        [6, 2, 1]
        >>> print(s1.tokens)
        ['<pad>', 'Ce@@', 'ci', 'est', 'un', 'te@@', 'st', '.', '<eos>']
        >>> s2 = Sentence(system_id= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> s2.suppr_pad(strict=True)
        [6, 2, 1, 0]
        >>> print(s2.tokens)
        ['Ce@@', 'ci', 'est', 'un', 'te@@', 'st', '.', '<eos>']
        """
        # Si None on calcule la liste des index à supprimer sinon on utilise celle fournie en argument
        list_index = Sentence.list_suppr_pad(self.tokens, padding_mark=padding_mark, strict=strict) if list_index is None else list_index

        # Suppression des tokens aux index spécifiés
        for i in list_index:
            del self.tokens[i]

        # Retour de la liste des index supprimés
        return list_index

    def fusion_bpe(self, list_groupes_bpe: List[List[int]] | None = None, BPE_mark: str = '@@') -> List[List[int]]:
        """Fusionne les tokens BPEisés et retourne la liste des groupes de tokens à fusionner dans l'ordre décroissant

        Args:
            list_bpe (List[List[int]]): Liste des indexs décroissant des tokens contenant une marque de BPE.
        
        Returns:
            List[List[int]]: Liste des groupes de tokens fusionnés. Liste décroissante contenant les groupes de tokens à fusionner dans l'ordre décroissant.

        Example:
        >>> snt3 = Sentence(system_id=2, tokens=["lu@@", "bu@@", "lu@@", "le", ".", "<eos>"])
        >>> snt3.fusion_bpe()
        [[5], [4], [3, 2, 1, 0]]
        >>> print(snt3)
        {'_system_id': 2, '_tokens': ['lubulule', '.', '<eos>']}
        """
        groupes_bpe = Sentence.list_fusion_bpe(self.tokens, BPE_mark=BPE_mark) if list_groupes_bpe is None else list_groupes_bpe
        
        if groupes_bpe is not None:
            # Cas particulier où il n'y a pas de bpe dans la phrase
            for groupe in groupes_bpe:
                if len(groupe) > 1:
                    for i in range(groupe[1], groupe[-1] -1 , -1):
                        self.tokens[i] = f"{self.tokens[i].split(BPE_mark)[0]}{self.tokens[i+1]}"
                        del self.tokens[i+1]

        return groupes_bpe

if __name__ == "__main__":
    import doctest;doctest.testmod()
    