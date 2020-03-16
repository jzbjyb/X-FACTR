from typing import Dict, Tuple, Set
import unicodedata as ud
import functools
from overrides import overrides
import unicodedata
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
    GENDER_MAP = {
        'male': 'MASC',
        'female': 'FEM',
        'none': 'NEUT'
    }


    def __init__(self,
                 entity2lang: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        self.entity2lang = entity2lang
        self.entity2instance = entity2instance
        self.disable_inflection: Set[str] = set(disable_inflection) if disable_inflection is not None else set()
        self.disable_article = disable_article


    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label


    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        if num_mask <= 0:
            return prompt.replace('[Y]', label), label
        return prompt.replace('[Y]', ' '.join([mask_sym] * num_mask)), label


    @staticmethod
    def from_lang(lang: str, *args, **kwargs):
        if lang == 'el':
            return PromptEL(*args, **kwargs)
        if lang == 'ru':
            return PromptRU(*args, **kwargs)
        return Prompt(*args, **kwargs)


class PromptEL(Prompt):
    SUFS = {'β', 'γ', 'δ', 'ζ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'τ', 'φ', 'χ', 'ψ'}


    def __init__(self,
                 entity2lang: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=False,
                 disable_article: bool=False):
        super().__init__(entity2lang=entity2lang,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)
        self.article: Dict[str, str] = {}
        with open('data/lang_resource/el/articles.txt') as inp:
            for l in inp:
                l = l.strip().split('\t')
                self.article[l[1]] = l[0]


    def gender_heuristic(self, w: str):
        w = w.strip()
        if ' ' not in w:
            if w[-1] in {"η", "ή", "α"}:
                return "FEM"
            elif w[-2:] in {"ος", "ης", "ής", "ας", "άς", "ός", "ήλ"}:
                return "MASC"
            else:
                return "NEUT"
        else:
            w2 = w.split(' ')[0]
            if w2[-1] in {"η", "ή", "α"}:
                return "FEM"
            elif w2[-2:] in {"ος", "ης", "ής", "ας", "άς", "ός", "ήλ"}:
                return "MASC"
            else:
                return "NEUT"


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = "SG"
        gender = self.GENDER_MAP[self.entity2lang[uri]]

        if gender == 'NEUT':  # use heuristics
            gender = self.gender_heuristic(label)
        if gender == 'NEUT' and uri in self.entity2instance:
            if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                last_char = label[-1]
                if last_char not in {'ν', 'ο', 'ό'}:
                    gender = "FEM"
            elif 'business' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'enterprise' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'city' in self.entity2instance[uri]:
                last_char = label[-1]
                if last_char not in {'ι', 'ο'}:
                    gender = "FEM"
            elif 'human' in self.entity2instance[uri]:
                gender = "MASC"
            elif 'island' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'literary work' in self.entity2instance[uri]:
                gender = "NEUT"
            elif 'musical group' in self.entity2instance[uri]:
                gender = "MASC"
                number = "PL"
            elif 'record label' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'language' in self.entity2instance[uri]:
                gender = "NEUT"
                number = "PL"
            elif 'sports team' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'automobile manufacturer' in self.entity2instance[uri]:
                gender = "FEM"
            elif 'football club' in self.entity2instance[uri]:
                gender = "FEM"

        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        if label[-2:] == "ες":
            ent_number = "PL"
            gender = "FEM"

        if some_roman_chars(label) or label.isupper() or label[-1] in self.SUFS:
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'x' in self.disable_inflection:
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
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        if label[-2:] in {"ες", "ές"}:
            ent_number = "PL"
            gender = "FEM"

        mask_sym = ' '.join([mask_sym] * num_mask)

        if some_roman_chars(label) or label.isupper() or label[-1] in self.SUFS:
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'y' in self.disable_inflection:
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


    def normalize(self, text: str, mask_sym: str='[MASK]') -> str:  # strip accents andlowercase
        nt = ''.join(c for c in unicodedata.normalize('NFD', text)
                     if unicodedata.category(c) != 'Mn').lower()
        return nt.replace(mask_sym.lower(), mask_sym)  # make sure the mask token is unchanged


class PromptRU(Prompt):
    SUFS = {"б", "в", "г", "д", "ж", "з", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"}


    # Decide on Russian gender for the unknown entities based on the endings
    # Based on http://www.study-languages-online.com/russian-nouns-gender.html
    @staticmethod
    def gender_heuristic(w):
        w = w.strip()
        if w[-1] == "a" or w[-1] == "я":
            return "FEM"
        elif w[-1] == "о" or w[-1] == "е" or w[-1] == "ё":
            return "NEUT"
        elif w[-1] == "й" or w[-1] in PromptRU.SUFS:
            return "MASC"
        else:
            return "MASC"


    def __init__(self,
                 entity2lang: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=False,
                 disable_article: bool=False):
        super().__init__(entity2lang=entity2lang,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender = self.entity2lang[uri]
        gender = self.GENDER_MAP[gender] if gender != 'none' else self.gender_heuristic(label).upper()
        ent_number = "SG"

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'x' in self.disable_inflection:
            do_not_inflect = True

        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            ent_case = "NOM"
        elif "[X.Nom]" in words:
            i = words.index('[X.Nom]')
            ent_case = "NOM"
        elif "[X.Masc.Nom]" in words:
            i = words.index('[X.Masc.Nom]')
            ent_case = "NOM"
            gender = "MASC"
        elif "[X.Gen]" in words:
            i = words.index('[X.Gen]')
            ent_case = "GEN"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='rus')[0]
        elif "[X.Ess]" in words:
            i = words.index('[X.Ess]')
            ent_case = "ESS"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ESS;{ent_number}", language='rus')[0]
        else:
            raise Exception('no X')
        words[i] = label

        # Now also check the correponsing articles, if the exist
        for i, w in enumerate(words):
            if w[0] == '[' and 'X-Gender' in w:
                if '|' in w:
                    option = w.strip()[1:-1].split('|')
                    if gender == "MASC" or gender == "FEM":
                        form = option[0].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = option[1].strip().split(';')[0]
                        words[i] = form
                else:
                    lemma = w.strip()[1:-1].split('.')[0]
                    form2 = lemma
                    if not self.disable_inflection:
                        if "Pst" in w:
                            form2 = cache_inflect(lemma, f"V;PST;SG;{gender}", language='rus')[0]
                        elif "Lgspec1" in w:
                            form2 = cache_inflect(lemma, f"ADJ;{gender};SG;LGSPEC1", language='rus')[0]
                    words[i] = form2

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender = self.entity2lang[uri]
        gender = self.GENDER_MAP[gender] if gender != 'none' else self.gender_heuristic(label).upper()
        ent_number = "SG"

        mask_sym = ' '.join([mask_sym] * num_mask)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'x' in self.disable_inflection:
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
            ent_case = "GEN"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='rus')[0]
        elif "[Y.Acc]" in words:
            i = words.index('[Y.Acc]')
            ent_case = "ACC"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ACC;{ent_number}", language='rus')[0]
        elif "[Y.Dat]" in words:
            i = words.index('[Y.Dat]')
            ent_case = "DAT"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;DAT;{ent_number}", language='rus')[0]
        elif "[Y.Ess]" in words:
            i = words.index('[Y.Ess]')
            ent_case = "ESS"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ESS;{ent_number}", language='rus')[0]
        elif "[Y.Ins]" in words:
            i = words.index('[Y.Ins]')
            ent_case = "INS"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;INS;{ent_number}", language='rus')[0]
        else:
            raise Exception('no Y')

        if num_mask <= 0:
            words[i] = label
        else:
            words[i] = mask_sym

        # Now also check the correponsing articles, if the exist
        for i, w in enumerate(words):
            if w[0] == '[' and 'Y-Gender' in w:
                lemma = w.strip()[1:-1].split('.')[0]
                form2 = lemma
                if not self.disable_inflection:
                    if "Pst" in w:
                        form2 = cache_inflect(lemma, f"V;PST;SG;{gender}", language='rus')[0]
                    elif "Lgspec1" in w:
                        form2 = cache_inflect(lemma, f"ADJ;{gender};SG;LGSPEC1", language='rus')[0]
                words[i] = form2

        return ' '.join(words), label
