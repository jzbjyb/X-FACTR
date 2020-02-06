from typing import Dict, Tuple
from collections import defaultdict
import unicodedata as ud
import functools
from unimorph_inflect import inflect
from check_gender import Gender, load_entity_gender


@functools.lru_cache(maxsize=None)
def cache_inflect(*args, **kwargs):
    return inflect(*args, **kwargs)


# Two functions needed to check if the entity has latin characters
# in which case we shouldn't inflect it in Greek
latin_letters = {}
def is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def some_roman_chars(unistr):
    return any(is_latin(uchr) for uchr in unistr if uchr.isalpha())  # isalpha suggested by John Machin


class Prompt(object):
    def fill_x(self, prompt: str, uri: str, label: str, gender: Gender=None) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label


    def fill_y(self, prompt: str, uri: str, label: str, gender: Gender=None,
               num_mask: int=1, mask_sym: str='[MASK]') -> Tuple[str, str]:
        if num_mask <= 0:
            return prompt.replace('[Y]', label), label
        return prompt.replace('[Y]', ' '.join(['[MASK]'] * num_mask)), label


    @staticmethod
    def from_lang(lang: str):
        if lang == 'el':
            return PromptEL()
        return Prompt()


class PromptEL(Prompt):
    SUFS = {'β', 'γ', 'δ', 'ζ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'τ', 'φ', 'χ', 'ψ'}
    GENDER_MAP = {
        'male': 'MASC',
        'female': 'FEM',
        'none': 'NEUT'
    }


    def __init__(self):
        super(PromptEL).__init__()
        self.article: Dict[str, str] = {}
        with open('data/lang_resource/el/articles.txt') as inp:
            for l in inp:
                l = l.strip().split('\t')
                self.article[l[1]] = l[0]


    def fill_x(self, prompt: str, uri: str, label: str, gender: Gender=None) -> Tuple[str, str]:
        gender = self.GENDER_MAP[gender]
        ent_number = "SG"
        if label[-2:] == "ες":
            ent_number = "PL"
            gender = "FEM"
        elif label[-1] == "ά":
            ent_number = "PL"
            gender = "NEUT"

        if some_roman_chars(label) or label.isupper() or label[-1] in self.SUFS:
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.Nom]" in words:
            # In Greek the default case is Nominative so we don't need to try to inflect it
            i = words.index('[X.Nom]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.Gen]" in words:
            i = words.index('[X.Gen]')
            if do_not_inflect:
                words[i] = label
            else:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='ell2')[0]
                words[i] = label
            ent_case = "GEN"
        elif "[X.Acc]" in words:
            i = words.index('[X.Acc]')
            if do_not_inflect:
                words[i] = label
            else:
                label = cache_inflect(label, f"N;ACC;{ent_number}", language='ell2')[0]
                words[i] = label
            ent_case = "ACC"

        # Now also check the correponsing articles, if the exist
        if "[DEF;X]" in words:
            i = words.index('[DEF;X]')
            words[i] = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"]
        if "[DEF.Gen;X]" in words:
            i = words.index('[DEF.Gen;X]')
            words[i] = self.article[f"ART;DEF;{gender};{ent_number};GEN"]
        if "[PREPDEF;X]" in words:
            i = words.index('[PREPDEF;X]')
            words[i] = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"]

        return ' '.join(words), label


    def fill_y(self, prompt: str, uri: str, label: str, gender: Gender=None,
               num_mask: int=1, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender = self.GENDER_MAP[gender]
        ent_number = "SG"
        if label[-2:] == "ες":
            ent_number = "PL"
            gender = "FEM"
        elif label[-1] == "ά":
            ent_number = "PL"
            gender = "NEUT"

        mask_sym = ' '.join([mask_sym] * num_mask)

        if some_roman_chars(label) or label.isupper() or label[-1] in self.SUFS:
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[Y]' in words:
            i = words.index('[Y]')
            ent_case = "NOM"
        elif "[Y.Nom]" in words:
            # In Greek the default case is Nominative so we don't need to try to inflect it
            i = words.index('[Y.Nom]')
            ent_case = "NOM"
        elif "[Y.Gen]" in words:
            i = words.index('[Y.Gen]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='ell2')[0]
            ent_case = "GEN"
        elif "[Y.Acc]" in words:
            i = words.index('[Y.Acc]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ACC;{ent_number}", language='ell2')[0]
            ent_case = "ACC"

        if num_mask <= 0:
            words[i] = label
        else:
            words[i] = mask_sym

        # Now also check the correponsing articles, if they exist
        if "[DEF;Y]" in words:
            i = words.index('[DEF;Y]')
            words[i] = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"]
        if "[DEF.Gen;Y]" in words:
            i = words.index('[DEF.Gen;Y]')
            words[i] = self.article[f"ART;DEF;{gender};{ent_number};GEN"]
        if "[PREPDEF;Y]" in words:
            i = words.index('[PREPDEF;Y]')
            words[i] = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"]
        if "[INDEF;Y]" in words:
            i = words.index('[INDEF;Y]')
            # print(f"ART;INDEF;{ent_gender};{ent_number};{ent_case}")
            # print(article[f"ART;INDEF;{ent_gender};{ent_number};{ent_case}"])
            words[i] = self.article[f"ART;INDEF;{gender};{ent_number};{ent_case}"]

        return ' '.join(words), label
