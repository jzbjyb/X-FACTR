from typing import Dict, Tuple
from collections import defaultdict
import unicodedata as ud
import functools
from overrides import overrides
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
    def __init__(self, disable_inflection: bool=False, disable_article: bool=False):
        self.disable_inflection = disable_inflection
        self.disable_article = disable_article


    def fill_x(self, prompt: str, uri: str, label: str, gender: Gender=None) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label


    def fill_y(self, prompt: str, uri: str, label: str, gender: Gender=None,
               num_mask: int=1, mask_sym: str='[MASK]') -> Tuple[str, str]:
        if num_mask <= 0:
            return prompt.replace('[Y]', label), label
        return prompt.replace('[Y]', ' '.join(['[MASK]'] * num_mask)), label


    @staticmethod
    def from_lang(lang: str, *args, **kwargs):
        if lang == 'el':
            return PromptEL(*args, **kwargs)
        return Prompt(*args, **kwargs)


class PromptEL(Prompt):
    SUFS = {'β', 'γ', 'δ', 'ζ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'τ', 'φ', 'χ', 'ψ'}
    GENDER_MAP = {
        'male': 'MASC',
        'female': 'FEM',
        'none': 'NEUT'
    }


    def __init__(self, disable_inflection: bool=False, disable_article: bool=False):
        super().__init__(disable_inflection=disable_inflection, disable_article=disable_article)
        self.article: Dict[str, str] = {}
        with open('data/lang_resource/el/articles.txt') as inp:
            for l in inp:
                l = l.strip().split('\t')
                self.article[l[1]] = l[0]


    @overrides
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

        if self.disable_inflection:
            do_not_inflect = True

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
        has_article = False
        if "[DEF;X]" in words:
            has_article = True
            i = words.index('[DEF;X]')
            art = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"]
        if "[DEF.Gen;X]" in words:
            has_article = True
            i = words.index('[DEF.Gen;X]')
            art = self.article[f"ART;DEF;{gender};{ent_number};GEN"]
        if "[PREPDEF;X]" in words:
            has_article = True
            i = words.index('[PREPDEF;X]')
            art = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"]

        if has_article:
            if self.disable_article:
                words[i] = ''
            else:
                words[i] = art

        return ' '.join(words), label


    @overrides
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

        if self.disable_inflection:
            do_not_inflect = True

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
        has_article = False
        if "[DEF;Y]" in words:
            has_article = True
            i = words.index('[DEF;Y]')
            art = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"]
        if "[DEF.Gen;Y]" in words:
            has_article = True
            i = words.index('[DEF.Gen;Y]')
            art = self.article[f"ART;DEF;{gender};{ent_number};GEN"]
        if "[PREPDEF;Y]" in words:
            has_article = True
            i = words.index('[PREPDEF;Y]')
            art = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"]
        if "[INDEF;Y]" in words:
            has_article = True
            i = words.index('[INDEF;Y]')
            # print(f"ART;INDEF;{ent_gender};{ent_number};{ent_case}")
            # print(article[f"ART;INDEF;{ent_gender};{ent_number};{ent_case}"])
            art = self.article[f"ART;INDEF;{gender};{ent_number};{ent_case}"]
        if "[DEF;Y.Fem]" in words:
            has_article = True
            i = words.index('[DEF;Y.Fem]')
            art = self.article[f"ART;DEF;FEM;{ent_number}"]

        if has_article:
            if self.disable_article:
                words[i] = ''
            else:
                words[i] = art

        return ' '.join(words), label
