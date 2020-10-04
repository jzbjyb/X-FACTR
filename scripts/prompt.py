from typing import Dict, Tuple, Set, List
import unicodedata as ud
from overrides import overrides
from joblib import Memory
import json
import unicodedata
from unimorph_inflect import inflect
from check_gender import Gender, load_entity_gender
memory = Memory('inflection_cache', verbose=0)


def persist_to_file(file_name: str, save_per_count: int=1):
    def decorator(original_func):
        cache: Dict[int, List[str]] = {}
        try:
            with open(file_name, 'r') as fin:
                for l in fin:
                    k, v = l.strip('\n').split('\t')
                    cache[int(k)] = json.loads(v)
        except (IOError, ValueError):
            pass
        delta: List[Tuple[int, List[str]]] = []
        def new_func(*args, **kwargs):
            param = hash(tuple(args) + tuple(sorted(kwargs.items())))
            if param not in cache:
                result = original_func(*args, **kwargs)
                cache[param] = result
                delta.append((param, result))
                if len(delta) >= save_per_count:
                    print('save {}'.format(len(cache)))
                    with open(file_name, 'a') as fout:
                        for k, v in delta:
                            fout.write('{}\t{}\n'.format(k, json.dumps(v)))
                    delta.clear()
            return cache[param]
        return new_func
    return decorator


#@functools.lru_cache(maxsize=None)
#@persist_to_file('inflection_cache.json', save_per_count=50)
@memory.cache
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
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        self.entity2gender = entity2gender
        self.entity2instance = entity2instance
        self.disable_inflection: Set[str] = set(disable_inflection) if disable_inflection is not None else set()
        self.disable_article = disable_article


    def get_target(self, label: str, num_mask: int, mask_sym: str) -> str:
        return label if num_mask <= 0 else ' '.join([mask_sym] * num_mask)


    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        return prompt.replace('[X]', label), label


    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        return prompt.replace('[Y]', self.get_target(label, num_mask, mask_sym)), label


    @staticmethod
    def from_lang(lang: str, *args, **kwargs):
        if lang == 'el':
            return PromptEL(*args, **kwargs)
        elif lang == 'ru':
            return PromptRU(*args, **kwargs)
        elif lang == 'fr':
            return PromptFR(*args, **kwargs)
        elif lang == 'es':
            return PromptES(*args, **kwargs)
        elif lang == 'mr':
            return PromptMR(*args, **kwargs)
        elif lang == 'he':
            return PromptHE(*args, **kwargs)
        elif lang == 'tr':
            return PromptTR(*args, **kwargs)
        elif lang == 'hu':
            return PromptHU(*args, **kwargs)
        elif lang == 'bn':
            return PromptBN(*args, **kwargs)
        return Prompt(*args, **kwargs)


class PromptEL(Prompt):
    SUFS = {'β', 'γ', 'δ', 'ζ', 'κ', 'λ', 'μ', 'ν', 'ξ', 'π', 'ρ', 'τ', 'φ', 'χ', 'ψ'}


    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)
        self.article: Dict[str, str] = {}
        with open('data/lang_resource/el/articles.txt') as inp:
            for l in inp:
                l = l.strip().split('\t')
                self.article[l[1]] = l[0]


    @staticmethod
    def gender_heuristic(w: str):
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
        gender = self.GENDER_MAP[self.entity2gender[uri]]

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

        try:
            # Now also check the correponsing articles, if the exist
            if "[DEF;X]" in words:
                i = words.index('[DEF;X]')
                words[i] = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
            if "[DEF.Gen;X]" in words:
                i = words.index('[DEF.Gen;X]')
                words[i] = self.article[f"ART;DEF;{gender};{ent_number};GEN"] \
                    if not self.disable_article else ''
            if "[PREPDEF;X]" in words:
                i = words.index('[PREPDEF;X]')
                words[i] = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
        except KeyError as e:
            print('article key error with prompt "{}", uri {}, label "{}", gender {}, number {}, case {}'.format(
                prompt, uri, label, gender, ent_number, ent_case))
            raise e

        # Now also check the corresponfing verbs, if they exist.
        # Needed for subject-verb agreement
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'X-Number' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_number == "SG":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[1].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        if label[-2:] in {"ες", "ές"}:
            ent_number = "PL"
            gender = "FEM"

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

        words[i] = self.get_target(label, num_mask, mask_sym)

        try:
            # Now also check the correponsing articles, if they exist
            if "[DEF;Y]" in words:
                i = words.index('[DEF;Y]')
                words[i] = self.article[f"ART;DEF;{gender};{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
            if "[DEF.Gen;Y]" in words:
                i = words.index('[DEF.Gen;Y]')
                words[i] = self.article[f"ART;DEF;{gender};{ent_number};GEN"] \
                    if not self.disable_article else ''
            if "[PREPDEF;Y]" in words:
                i = words.index('[PREPDEF;Y]')
                words[i] = self.article[f"ART;PREPDEF;{gender};{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
            if "[INDEF;Y]" in words:
                i = words.index('[INDEF;Y]')
                words[i] = self.article[f"ART;INDEF;{gender};{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
            if "[DEF;Y.Fem]" in words:
                i = words.index('[DEF;Y.Fem]')
                words[i] = self.article[f"ART;DEF;FEM;{ent_number};{ent_case}"] \
                    if not self.disable_article else ''
        except KeyError as e:
            print('article key error with prompt "{}", uri {}, label "{}", gender {}, number {}, case {}'.format(
                prompt, uri, label, gender, ent_number, ent_case))
            raise e

        return ' '.join(words), label


    def normalize(self, text: str, mask_sym: str='[MASK]') -> str:  # strip accents andlowercase
        nt = ''.join(c for c in unicodedata.normalize('NFD', text)
                     if unicodedata.category(c) != 'Mn').lower()
        return nt.replace(mask_sym.lower(), mask_sym)  # make sure the mask token is unchanged


class PromptRU(Prompt):
    SUFS = {"й", "б", "в", "г", "д", "ж", "з", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"}


    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    # Decide on Russian gender for the unknown entities based on the endings
    # Based on http://www.study-languages-online.com/russian-nouns-gender.html
    @staticmethod
    def gender_heuristic(w):
        w = w.strip()
        if w[-1] in {"a", "я"}:
            return "FEM"
        elif w[-1] in {"о", "е", "ё"}:
            return "NEUT"
        elif w[-1] in PromptRU.SUFS:
            return "MASC"
        else:
            return "MASC"


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = "SG"
        gender = self.GENDER_MAP[self.entity2gender[uri]]

        if gender == 'NEUT':  # use heuristics
            gender = self.gender_heuristic(label)
        if gender == 'NEUT' and uri in self.entity2instance:
            if 'human' in self.entity2instance[uri]:
                gender = "MASC"

        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)

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
            words[i] = label
        elif "[X.Nom]" in words:
            i = words.index('[X.Nom]')
            ent_case = "NOM"
            words[i] = label
        elif "[X.Masc.Nom]" in words:
            i = words.index('[X.Masc.Nom]')
            ent_case = "NOM"
            gender = "MASC"
            words[i] = label
        elif "[X.Gen]" in words:
            i = words.index('[X.Gen]')
            ent_case = "GEN"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='rus')[0]
            words[i] = label
        elif "[X.Ess]" in words:
            i = words.index('[X.Ess]')
            ent_case = "ESS"
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ESS;{ent_number}", language='rus')[0]
            words[i] = label
        else:
            raise Exception('no X')

        # Now also check the correponsing verbs, if they exist
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'X-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif gender == "FEM":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    elif gender == "NEUT":
                        form = options[2].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
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
        gender, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
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

        words[i] = self.get_target(label, num_mask, mask_sym)

        # Now also check the correponsing articles, if the exist
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'Y-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif gender == "FEM":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    elif gender == "NEUT":
                        form = options[2].strip().split(';')[0]
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


class PromptFR(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)
        self.article: Dict[str, str] = {}
        with open('data/lang_resource/fr/articles.txt') as inp:
            for l in inp:
                l = l.strip().split('\t')
                self.article[l[1]] = l[0]

    @staticmethod
    def starts_with_vowel(w):
        w = w.strip()
        if w[0] in {'a', 'o', 'i', 'e', 'u', 'y', 'à', 'è', 'ù', 'é', 'â', 'ê', 'î', 'ô', 'û'}:
            return True
        return False


    @staticmethod
    def gender_heuristic(w):
        # Based on this: https://frenchtogether.com/french-nouns-gender/
        w = w.strip()
        if ' ' not in w:
            if w[-6:] in {"ouille"}:
                return "FEM"
            elif w[-5:] in {"aisse", "ousse", "aille", "eille", "ouche", "anche"}:
                return "FEM"
            elif w[-4:] in {"asse", "esse", "isse", "ance", "anse", "ence", "once", "enne", "onne", "aine", "eine",
                            "erne", "ande", "ende", "onde", "arde", "orde", "euse", "ouse", "aise", "oise", "ache",
                            "iche", "ehce", "oche", "uche", "iere", "eure", "atte", "otte", "oute", "orte", "ante",
                            "ente", "inte", "onte", "alle", "elle", "ille", "olle", "appe", "ampe", "ombe", "igue"}:
                return "FEM"
            elif w[-4:] in {"aume", "isme", "ours", "euil", "ueil"}:
                return "MASC"
            elif w[-3:] in {"aie", "oue", "eue", "ace", "ece", "ice", "une", "ine", "ade", "ase", "ese", "ise", "yse",
                            "ose", "use", "ave", "eve", "ive", "ete", "ête"}:
                return "FEM"
            elif w[-3:] in {"and", "ant", "ent", "int", "ond", "ont", "eau", "aud", "aut", "ais", "ait", "out", "oux",
                            "age", "ege", "ème", "ome", "est", "eul", "all", "air", "erf", "ert", "arc", "ars", "art",
                            "our", "ord", "ors", "ort", "oir", "eur", "ail", "eil", "ing"}:
                return "MASC"
            elif w[-2:] in {"te", "ée", "ie", "ue"}:
                return "FEM"
            elif w[-2:] in {"an", "in", "om", "on", "au", "os", "ot", "ai", "es", "et", "ou", "il", "it", "is", "at",
                            "as", "us", "ex", "al", "el", "ol", "if", "ef", "ac", "ic", "oc", "uc", "um", "am", "en"}:
                return "MASC"
            elif w[-1] in {"o", "i", "y", "u"}:
                return "MASC"
            else:
                return "MASC"
            # ARGH
        else:
            w2 = w.split(' ')[0]
            if w2[-6:] in {"ouille"}:
                return "FEM"
            elif w2[-5:] in {"aisse", "ousse", "aille", "eille", "ouche", "anche"}:
                return "FEM"
            elif w2[-4:] in {"asse", "esse", "isse", "ance", "anse", "ence", "once", "enne", "onne", "aine", "eine",
                             "erne", "ande", "ende", "onde", "arde", "orde", "euse", "ouse", "aise", "oise", "ache",
                             "iche", "ehce", "oche", "uche", "iere", "eure", "atte", "otte", "oute", "orte", "ante",
                             "ente", "inte", "onte", "alle", "elle", "ille", "olle", "appe", "ampe", "ombe", "igue"}:
                return "FEM"
            elif w2[-4:] in {"aume", "isme", "ours", "euil", "ueil"}:
                return "MASC"
            elif w2[-3:] in {"aie", "oue", "eue", "ace", "ece", "ice", "une", "ine", "ade", "ase", "ese", "ise", "yse",
                             "ose", "use", "ave", "eve", "ive", "ete", "ête"}:
                return "FEM"
            elif w2[-3:] in {"and", "ant", "ent", "int", "ond", "ont", "eau", "aud", "aut", "ais", "ait", "out", "oux",
                             "age", "ege", "ème", "ome", "est", "eul", "all", "air", "erf", "ert", "arc", "ars", "art",
                             "our", "ord", "ors", "ort", "oir", "eur", "ail", "eil", "ing"}:
                return "MASC"
            elif w2[-2:] in {"te", "ée", "ie", "ue"}:
                return "FEM"
            elif w2[-2:] in {"an", "in", "om", "on", "au", "os", "ot", "ai", "es", "et", "ou", "il", "it", "is", "at",
                             "as", "us", "ex", "al", "el", "ol", "if", "ef", "ac", "ic", "oc", "uc", "um", "am", "en"}:
                return "MASC"
            elif w2[-1] in {"o", "i", "y", "u"}:
                return "MASC"
            else:
                return "MASC"
            # ARGH


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str, bool, bool, bool]:
        number = "SG"
        country = False
        city = False
        proper = True
        gender = self.GENDER_MAP[self.entity2gender[uri]]

        if gender == 'NEUT':  # use heuristics
            gender = self.gender_heuristic(label)
            if uri in self.entity2instance:
                if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                    country = True
                    proper = True
                elif 'business' in self.entity2instance[uri]:
                    gender = "FEM"
                    proper = True
                elif 'enterprise' in self.entity2instance[uri]:
                    gender = "FEM"
                    proper = True
                elif 'city' in self.entity2instance[uri]:
                    city = True
                    proper = True
                elif 'human' in self.entity2instance[uri]:
                    gender = "MASC"
                elif 'island' in self.entity2instance[uri]:
                    proper = True
                elif 'musical group' in self.entity2instance[uri]:
                    proper = True
                elif 'record label' in self.entity2instance[uri]:
                    proper = True
                elif 'language' in self.entity2instance[uri]:
                    gender = "MASC"
                elif 'sports team' in self.entity2instance[uri]:
                    gender = "FEM"
                    proper = True
                elif 'automobile manufacturer' in self.entity2instance[uri]:
                    gender = "FEM"
                    proper = True
                elif 'football club' in self.entity2instance[uri]:
                    gender = "FEM"
                    proper = True

        return gender, number, country, city, proper


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        ent_gender, ent_number, ent_country, ent_city, ent_proper = self.get_ender_number(uri, label)

        if label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label

        # Now also check the correponding articles, if they exist
        if "[ARTDEF;X]" in words:
            has_article: bool = False
            i = words.index('[ARTDEF;X]')
            vowel = self.starts_with_vowel(words[i + 1])
            if ent_proper and not ent_country:
                del words[i]
            elif vowel:
                has_article = True
                art = self.article["ARTDEF;VOWEL"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"ARTDEF;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"ARTDEF;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[PREPDEF;X]" in words:
            has_article: bool = False
            i = words.index('[PREPDEF;X]')
            if ent_proper and not ent_country:
                has_article = True
                art = self.article[f"PREPDEF;PROPN"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"PREPDEF;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"PREPDEF;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[PREPDEF-à;X]" in words:
            has_article: bool = False
            i = words.index('[PREPDEF-à;X]')
            if ent_proper and not ent_country:
                has_article = True
                art = self.article[f"PREPDEF-à;PROPN"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"PREPDEF-à;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"PREPDEF-à;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art

        # Now also check the corresponfing verbs, if they exist.
        # Needed for subject-verb agreement
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'X-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[1].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
              num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        ent_gender, ent_number, ent_country, ent_city, ent_proper = self.get_ender_number(uri, label)

        if label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[Y]' in words:
            i = words.index('[Y]')
            words[i] = self.get_target(label, num_mask, mask_sym)

        # Now also check the correponsing articles, if they exist
        if "[ARTDEF;Y]" in words:
            has_article: bool = False
            i = words.index('[ARTDEF;Y]')
            vowel = self.starts_with_vowel(words[i + 1])
            if ent_proper and not ent_country:
                del words[i]
            elif vowel:
                has_article = True
                art = self.article["ARTDEF;VOWEL"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"ARTDEF;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"ARTDEF;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[ARTIND;Y]" in words:
            has_article: bool = False
            i = words.index('[ARTIND;Y]')
            if ent_number == "SG":
                has_article = True
                art = self.article[f"ARTIND;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"ARTIND;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[PREPDEF;Y]" in words:
            has_article: bool = False
            i = words.index('[PREPDEF;Y]')
            if ent_proper and not ent_country:
                has_article = True
                art = self.article[f"PREPDEF;PROPN"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"PREPDEF;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"PREPDEF;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[PREPDEF-à;Y]" in words:
            has_article: bool = False
            i = words.index('[PREPDEF-à;Y]')
            if ent_proper and not ent_country:
                has_article = True
                art = self.article[f"PREPDEF-à;PROPN"]
            elif ent_number == "SG":
                has_article = True
                art = self.article[f"PREPDEF-à;{ent_gender};{ent_number}"]
            else:
                has_article = True
                art = self.article[f"PREPDEF-à;{ent_number}"]
            if has_article:
                words[i] = '' if self.disable_article else art
        if "[PREPLOC;Y]" in words:
            i = words.index('[PREPLOC;Y]')
            if ent_city:
                has_article = True
                art = self.article[f"PREPLOC;CITY"]
            elif ent_country:
                has_article = True
                art = self.article[f"PREPLOC;COUNTRY"]
            else:
                has_article = True
                art = self.article[f"PREPLOC;COUNTRY"]
            if has_article:
                words[i] = '' if self.disable_article else art

        # Now also check the corresponfing verbs, if they exist.
        # Needed for subject-verb agreement
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'Y-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[1].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


class PromptES(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    @staticmethod
    def gender_heuristic(w):
        '''
        Decide on Spanish gender for the unknown entities based on the endings
        Based on http://www.study-languages-online.com/russian-nouns-gender.html
        '''
        w = w.strip()
        if ' ' not in w:
            if w[-1] == "o":
                return "MASC"
            elif w[-1] == "a" or w[-4:] in {"ción", "sión"} or w[-3:] in {"dad", "tad"} or w[-5:] in {"umbre"}:
                return "FEM"
            else:
                return "MASC"
        else:
            w2 = w.split(' ', 1)[0]
            if w2[-1] == "o":
                return "MASC"
            elif w2[-1] == "a" or w2[-4:] in {"ción", "sión"} or w2[-3:] in {"dad", "tad"} or w2[-5:] in {"umbre"}:
                return "FEM"
            else:
                return "MASC"


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = "SG"
        gender = self.GENDER_MAP[self.entity2gender[uri]]

        if gender == 'NEUT':  # use heuristics
            gender = self.gender_heuristic(label)
            if uri in self.entity2instance:
                # ARGH WHAT TO DO HERE : Using MASC because it is the most common one :(
                if 'human' in self.entity2instance[uri]:
                    gender = "MASC"
                if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                    gender = "FEM"
                elif 'musical group' in self.entity2instance[uri]:
                    gender = "MASC"
                    number = "PL"
                elif 'record label' in self.entity2instance[uri]:
                    gender = "FEM"
                elif 'football club' in self.entity2instance[uri]:
                    gender = "FEM"
        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        ent_gender, ent_number = self.get_ender_number(uri, label)

        if label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label

        if '[ART;X-Gender]' in words:
            i = words.index('[ART;X-Gender]')
            if ent_gender == "MASC":
                art = 'un'
            elif ent_gender == "FEM":
                art = 'una'
            else:
                art = 'un'
            words[i] = '' if self.disable_article else art

        # Now also check the corresponding verbs, if they exist
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[':
                if '|' in w and 'X-Gender' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_number == "PL" and len(options) == 3:
                        form = options[2].strip().split(';')[0]
                        words[i] = form
                    elif ent_gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif ent_gender == "FEM":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                elif '|' in w and 'X-Number' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_number == "PL":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    elif ent_number == "SG":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        ent_gender, ent_number = self.get_ender_number(uri, label)

        if label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[Y]' in words:
            i = words.index('[Y]')
            words[i] = self.get_target(label, num_mask, mask_sym)

        if '[ART;Y-Gender]' in words:
            i = words.index('[ART;Y-Gender]')
            if ent_gender == "MASC":
                art = 'un'
            elif ent_gender == "FEM":
                art = 'una'
            else:
                art = 'un'
            words[i] = '' if self.disable_article else art

        if '[DEF;Y]' in words:
            i = words.index('[DEF;Y]')
            if ent_gender == "MASC":
                art = 'el'
            elif ent_gender == "FEM":
                art = 'la'
            else:
                art = 'un'
            words[i] = '' if self.disable_article else art

        # Now also check the correponsing verb, if they exist
        for i, w in enumerate(words):
            if len(w) == 0:
                continue
            if len(w) > 0 and w[0] == '[' and 'Y-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_number == "PL" and len(options) == 3:
                        form = options[2].strip().split(';')[0]
                        words[i] = form
                    elif ent_gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif ent_gender == "FEM":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


class PromptMR(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    @staticmethod
    def gender_heuristic(w):
        w = w.strip()
        if ' ' not in w:
            return "NEUT"
        else:
            w2 = w.split(' ', 1)[0]
            return "NEUT"


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = 'SG'
        gender = self.GENDER_MAP[self.entity2gender[uri]]

        if gender == 'NEUT':  # use heuristics
            gender = self.gender_heuristic(label)
            if uri in self.entity2instance and gender == "NEUT":
                if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                    gender = "NEUT"
                elif 'human' in self.entity2instance[uri]:
                    # ARGH WHAT TO DO HERE
                    gender = "MASC"
        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        ent_gender, ent_number = self.get_ender_number(uri, label)
        if label[-2:] == "ες":
            ent_number = "PL"
            ent_gender = "FEM"

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.NOM]" in words:
            # In Greek the default case is Nominative so we don't need to try to inflect it
            i = words.index('[X.NOM]')
            words[i] = label
            ent_case = "NOM"

        # Now check for the ones that we have a fixed suffix:
        for i, w in enumerate(words):
            if w[:3] == '[X]' and len(w) > 3:
                words[i] = label + w[3:]

        # Now also check the corresponfing verbs, if they exist.
        # Needed for subject-verb agreement
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'X-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if ent_gender == "MASC":
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif ent_gender == "FEM":
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        ent_gender, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        words = prompt.split(' ')

        i: int = -1
        if '[Y]' in words:
            i = words.index('[Y]')
            ent_case = "NOM"
        elif "[Y.LOC]" in words:
            # In Greek the default case is Nominative so we don't need to try to inflect it
            i = words.index('[Y.LOC]')
            if not do_not_inflect:
                # words[i] = cache_inflect(ent_form, f"N;LOC;{ent_number}", language='mar')[0]
                label = label + 'त'
            ent_case = "LOC"

        if i != -1:
            words[i] = self.get_target(label, num_mask, mask_sym)

        # Now check for the ones that we have a fixed suffix:
        for i, w in enumerate(words):
            if w[:3] == '[Y]' and len(w) > 3:
                words[i] = self.get_target(label, num_mask, mask_sym) + w[3:]

        return ' '.join(words), label


class PromptTR(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)
        if disable_inflection or disable_article:
            raise NotImplementedError


    def fix_up(self, w, inds):
        out = ''
        for i, c in enumerate(w):
            if i in inds:
                out += c.upper()
            else:
                out += c
        return out

    def add_be(self, w, number):
        change = 'ç f h k p s ş t'.split()
        vowels = "a,o,u,ı,e,ü,ö,i".split(',')
        undotted = "a,o,u,ı".split(',')
        dotted = "e,ü,ö,i".split(',')

        last_vowel = None
        i = len(w) - 1
        while i >= 0:
            if w[i] in vowels:
                last_vowel = w[i]
                break
            else:
                i -= 1

        first = '\'d'
        if w[-1] in change:
            first = '\'t'

        end = first + 'ir'
        if number == "PL":
            end += "lar"

        if last_vowel in undotted:
            if last_vowel == "a" or last_vowel == 'ı':
                end = first + "ır"
                if number == "PL":
                    end += "lar"
            elif last_vowel == "o" or last_vowel == 'u':
                end = first + "ur"
                if number == "PL":
                    end += "lar"
        elif last_vowel in dotted:
            if last_vowel == "e" or last_vowel == 'i':
                end = first + "ir"
                if number == "PL":
                    end += "ler"
            elif last_vowel == "ö" or last_vowel == 'ü':
                end = first + "ür"
                if number == "PL":
                    end += "ler"

        return w + end


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = 'SG'
        gender = 'NEUT'
        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        words = prompt.split(' ')

        inds = [i for i, x in enumerate(label) if x.isupper()]
        if inds:
            label2 = label.lower()
        else:
            label2 = label

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.Loc]" in words:
            i = words.index('[X.Loc]')
            temp = cache_inflect(label2, f"N;LOC;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = label
            ent_case = "LOC"
        elif "[X.Gen]" in words:
            i = words.index('[X.Gen]')
            temp = cache_inflect(label2, f"N;GEN;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = label
            ent_case = "GEN"
        elif "[X.Acc]" in words:
            i = words.index('[X.Acc]')
            temp = cache_inflect(label2, f"N;ACC;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = label
            ent_case = "ACC"
        elif "[X.Dat]" in words:
            i = words.index('[X.Dat]')
            temp = cache_inflect(label2, f"N;DAT;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = label
            ent_case = "DAT"
        elif "[X.Abl]" in words:
            i = words.index('[X.Abl]')
            temp = cache_inflect(label2, f"N;ABL;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = label
            ent_case = "ABL"

        if '[X;be]' in words:
            i = words.index('[X;be]')
            label = self.add_be(label, ent_number)
            words[i] = label

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        words = prompt.split(' ')

        inds = [i for i, x in enumerate(label) if x.isupper()]
        if inds:
            label2 = label.lower()
        else:
            label2 = label

        if '[Y]' in words:
            i = words.index('[Y]')
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "NOM"
        elif "[Y.Loc]" in words:
            i = words.index('[Y.Loc]')
            temp = cache_inflect(label2, f"N;LOC;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "LOC"
        elif "[Y.Gen]" in words:
            i = words.index('[Y.Gen]')
            temp = cache_inflect(label2, f"N;GEN;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "GEN"
        elif "[Y.Acc]" in words:
            i = words.index('[Y.Acc]')
            temp = cache_inflect(label2, f"N;ACC;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "ACC"
        elif "[Y.Dat]" in words:
            i = words.index('[Y.Dat]')
            temp = cache_inflect(label2, f"N;DAT;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "DAT"
        elif "[Y.Abl]" in words:
            i = words.index('[Y.Abl]')
            temp = cache_inflect(label2, f"N;ABL;{ent_number}", language='tur')[0]
            if inds:
                label = self.fix_up(temp, inds)
            else:
                label = temp
            words[i] = self.get_target(label, num_mask, mask_sym)
            ent_case = "ABL"

        if '[Y;be]' in words:
            i = words.index('[Y;be]')
            label = self.add_be(label, ent_number)
            words[i] = self.get_target(label, num_mask, mask_sym)

        return ' '.join(words), label


class PromptHE(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str = None,
                 disable_article: bool = False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = "SG"
        gender = self.GENDER_MAP[self.entity2gender[uri]]

        if gender == 'NEUT' and uri in self.entity2instance:
            if 'human' in self.entity2instance[uri]:
                gender = 'MASC'

        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        words = prompt.split(' ')

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
        for i, w in enumerate(words):
            if '[X]' in w:
                form = w.replace('[X]', label)
                words[i] = form

        # Now also check the corresponfing verbs, if they exist
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'X-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if gender == 'MASC':
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif gender == 'FEM':
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int = 0, mask_sym: str = '[MASK]') -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)
        words = prompt.split(' ')

        if '[Y]' in words:
            i = words.index('[Y]')
            words[i] = self.get_target(label, num_mask, mask_sym)
        for i, w in enumerate(words):
            if '[Y]' in w:
                form = w.replace('[Y]', self.get_target(label, num_mask, mask_sym))
                words[i] = form

        # Now also check the correponsing articles, if the exist
        for i, w in enumerate(words):
            if len(w) > 0 and w[0] == '[' and 'Y-Gender' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if gender == 'MASC':
                        form = options[0].strip().split(';')[0]
                        words[i] = form
                    elif gender == 'FEM':
                        form = options[1].strip().split(';')[0]
                        words[i] = form
                    else:
                        form = options[0].strip().split(';')[0]
                        words[i] = form

        return ' '.join(words), label


class PromptHU(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = 'SG'
        gender = self.GENDER_MAP[self.entity2gender[uri]]
        is_human = False
        if gender == 'NEUT' and uri in self.entity2instance:
            if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                is_human = False
            elif 'human' in self.entity2instance[uri]:
                is_human = True
        return gender, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'x' in self.disable_inflection:
            do_not_inflect = True

        words = prompt.split(' ')
        words = [w.strip() for w in words if w.strip()]

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.ACC]" in words:
            i = words.index('[X.ACC]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ACC;{ent_number}", language='hun')[0]
            words[i] = label
            ent_case = "ACC"
        elif "[X.DAT]" in words:
            i = words.index('[X.DAT]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;DAT;{ent_number}", language='hun')[0]
            words[i] = label
            ent_case = "DAT"
        elif "[X.ON+ESS]" in words:
            i = words.index('[X.ON+ESS]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ON+ESS;{ent_number}", language='hun')[0]
            words[i] = label
            ent_case = "ON+ESS"
        else:
            raise Exception('no X')

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        gender, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'y' in self.disable_inflection:
            do_not_inflect = True

        words = prompt.split(' ')
        words = [w.strip() for w in words if w.strip()]

        i = -1
        if '[Y]' in words:
            i = words.index('[Y]')
            words[i] = label
            ent_case = "NOM"
        elif "[Y.IN+ESS]" in words:
            i = words.index('[Y.IN+ESS]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;IN+ESS;{ent_number}", language='hun')[0]
            ent_case = "IN+ESS"
        elif "[Y.IN+ABL]" in words:
            i = words.index('[Y.IN+ABL]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;IN+ABL;{ent_number}", language='hun')[0]
            ent_case = "IN+ABL"
        elif "[Y.ON+ESS]" in words:
            i = words.index('[Y.ON+ESS]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;ON+ESS;{ent_number}", language='hun')[0]
            ent_case = "ON+ESS"
        elif "[Y.DAT]" in words:
            i = words.index('[Y.DAT]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;DAT;{ent_number}", language='hun')[0]
            ent_case = "DAT"
        elif "[Y.ACC]" in words:
            i = words.index('[Y.ACC]')
            if do_not_inflect:
                label = cache_inflect(label, f"N;ACC;{ent_number}", language='hun')[0]
            ent_case = "ACC"
        elif "[Y.INST]" in words:
            i = words.index('[Y.INST]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;INST;{ent_number}", language='hun')[0]
            ent_case = "INST"

        if i != -1:
            words[i] = self.get_target(label, num_mask, mask_sym)

        # Now check for the ones that we have a fixed suffix:
        for i, w in enumerate(words):
            if w[:3] == '[Y]' and len(w) > 3:
                words[i] = label + w[3:]

        return ' '.join(words), label


class PromptBN(Prompt):
    def __init__(self,
                 entity2gender: Dict[str, Gender],
                 entity2instance: Dict[str, str],
                 disable_inflection: str=None,
                 disable_article: bool=False):
        super().__init__(entity2gender=entity2gender,
                         entity2instance=entity2instance,
                         disable_inflection=disable_inflection,
                         disable_article=disable_article)


    def get_ender_number(self, uri: str, label: str) -> Tuple[str, str]:
        number = 'SG'
        gender = self.GENDER_MAP[self.entity2gender[uri]]
        is_human = False
        if gender == 'NEUT' and uri in self.entity2instance:
            if 'state' in self.entity2instance[uri] or 'country' in self.entity2instance[uri]:
                is_human = False
            elif 'human' in self.entity2instance[uri]:
                is_human = True
        return is_human, number


    @overrides
    def fill_x(self, prompt: str, uri: str, label: str) -> Tuple[str, str]:
        is_human, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'x' in self.disable_inflection:
            do_not_inflect = True

        words = prompt.split(' ')
        words = [w.strip() for w in words if w.strip()]

        if '[X]' in words:
            i = words.index('[X]')
            words[i] = label
            ent_case = "NOM"
        elif "[X.LOC]" in words:
            i = words.index('[X.LOC]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;LOC;{ent_number}", language='ben')[0]
            words[i] = label
            ent_case = "LOC"
        elif "[X.GEN]" in words:
            i = words.index('[X.GEN]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='ben')[0]
            words[i] = label
            ent_case = "GEN"

        # Now check for the ones that we have a fixed suffix:
        for i, w in enumerate(words):
            if w[:3] == '[X]' and len(w) > 3:
                words[i] = label + w[3:]

        # Now also check the corresponfing verbs, if they exist.
        # Needed for subject-verb agreement
        for i, w in enumerate(words):
            if w[0] == '[' and 'human' in w:
                if '|' in w:
                    options = w.strip()[1:-1].split('|')
                    if is_human:
                        label = options[0].strip().split(':')[1]
                        words[i] = label
                    else:
                        label = options[1].strip().split(':')[1]
                        words[i] = label

        return ' '.join(words), label


    @overrides
    def fill_y(self, prompt: str, uri: str, label: str,
               num_mask: int=0, mask_sym: str='[MASK]') -> Tuple[str, str]:
        is_human, ent_number = self.get_ender_number(uri, label)

        if some_roman_chars(label) or label.isupper():
            do_not_inflect = True
        else:
            do_not_inflect = False

        if 'y' in self.disable_inflection:
            do_not_inflect = True

        words = prompt.split(' ')
        words = [w.strip() for w in words if w.strip()]

        i = -1
        if '[Y]' in words:
            i = words.index('[Y]')
            ent_case = "NOM"
        elif "[Y.LOC]" in words:
            i = words.index('[Y.LOC]')
            if not do_not_inflect:
               label = cache_inflect(label, f"N;LOC;{ent_number}", language='ben')[0]
            ent_case = "LOC"
        elif "[Y.GEN]" in words:
            i = words.index('[Y.GEN]')
            if not do_not_inflect:
                label = cache_inflect(label, f"N;GEN;{ent_number}", language='ben')[0]
            ent_case = "GEN"

        if i != -1:
            words[i] = self.get_target(label, num_mask, mask_sym)

        # Now check for the ones that we have a fixed suffix:
        for i, w in enumerate(words):
            if w[:3] == '[Y]' and len(w) > 3:
                words[i] = label + w[3:]

        return ' '.join(words), label
