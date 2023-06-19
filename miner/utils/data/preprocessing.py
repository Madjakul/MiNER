# miner/utils/data/preprocessing.py

import os
import re
from typing import Union, Optional

from spacy.tokens import Doc
from spacy.lang.fr import French
from spacy.lang.en import English


STOP_WORDS_FR = set(
    """
a à â abord afin ah ai aie ainsi ait allaient allons
alors anterieur anterieure anterieures antérieur antérieure antérieures
apres après as assez attendu au
aupres auquel aura auraient aurait auront
aussi autre autrement autres autrui aux auxquelles auxquels avaient
avais avait avant avec avoir avons ayant
bas basee bat
c' c’ ça car ce ceci cela celle celle-ci celle-la celle-là celles celles-ci celles-la celles-là
celui celui-ci celui-la celui-là cent cependant certain certaine certaines certains certes ces
cet cette ceux ceux-ci ceux-là chacun chacune chaque chez ci cinq cinquantaine cinquante
cinquantième cinquième combien comme comment compris concernant
d' d’ da dans de debout dedans dehors deja dejà delà depuis derriere
derrière des desormais desquelles desquels dessous dessus deux deuxième
deuxièmement devant devers devra different differente differentes differents différent
différente différentes différents dire directe directement dit dite dits divers
diverse diverses dix dix-huit dix-neuf dix-sept dixième doit doivent donc dont
douze douzième du duquel durant dès déja déjà désormais
effet egalement eh elle elle-meme elle-même elles elles-memes elles-mêmes en encore
enfin entre envers environ es ès est et etaient étaient etais étais etait était
etant étant etc etre être eu eux eux-mêmes exactement excepté également
fais faisaient faisant fait facon façon feront font
gens
ha hem hep hi ho hormis hors hou houp hue hui huit huitième
hé i il ils importe
j' j’ je jusqu jusque juste
l' l’ la laisser laquelle le lequel les lesquelles lesquels leur leurs longtemps
lors lorsque lui lui-meme lui-même là lès
m' m’ ma maint maintenant mais malgre malgré me meme memes merci mes mien
mienne miennes miens mille moi moi-meme moi-même moindres moins
mon même mêmes
n' n’ na ne neanmoins neuvième ni nombreuses nombreux nos notamment
notre nous nous-mêmes nouveau nul néanmoins nôtre nôtres
o ô on ont onze onzième or ou ouias ouste outre
ouvert ouverte ouverts où
par parce parfois parle parlent parler parmi partant
pas pendant pense permet personne peu peut peuvent peux plus
plusieurs plutot plutôt possible possibles pour pourquoi
pourrais pourrait pouvait prealable precisement
premier première premièrement
pres procedant proche près préalable précisement pu puis puisque
qu' qu’ quand quant quant-à-soi quarante quatorze quatre quatre-vingt
quatrième quatrièmement que quel quelconque quelle quelles quelqu'un quelque
quelques quels qui quiconque quinze quoi quoique
relative relativement rend rendre restant reste
restent retour revoici revoila revoilà
s' s’ sa sait sans sauf se seize selon semblable semblaient
semble semblent sent sept septième sera seraient serait seront ses seul seule
seulement seuls seules si sien sienne siennes siens sinon six sixième soi soi-meme soi-même soit
soixante son sont sous souvent specifique specifiques spécifique spécifiques stop
suffisant suffisante suffit suis suit suivant suivante
suivantes suivants suivre sur surtout
t' t’ ta tant te tel telle tellement telles tels tenant tend tenir tente
tes tien tienne tiennes tiens toi toi-meme toi-même ton touchant toujours tous
tout toute toutes treize trente tres trois troisième troisièmement très
tu té
un une unes uns
va vais vas vers via vingt voici voila voilà vont vos
votre votres vous vous-mêmes vu vé vôtre vôtres
y
""".split()
)


STOP_WORDS_EN = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at
back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by
call can cannot ca could
did do does doing done down due during
each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except
few fifteen fifty first five for former formerly forty four from front full
further
get give go
had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred
i if in indeed into is it its itself
keep
last latter latterly least less
just
made make many may me meanwhile might mine more moreover most mostly move much
must my myself
name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere
of off often on once one only onto or other others otherwise our ours ourselves
out over own
part per perhaps please put
quite
rather re really regarding
s same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such
take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two
under until up unless upon us used using
various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would
yet you your yours yourself yourselves
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
STOP_WORDS_EN.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        STOP_WORDS_EN.add(stopword.replace("'", apostrophe))


def load_gazetteers(path: str):
    """Loads every files in a given directory and treats each of them as a
    dictionary's key.

    Parameters
    ----------
    path: ``str``
        Path to the directory containing the dictionaries.

    Returns
    -------
    gazetteers: ``dict``
        Dictionary with format {"name_of_file": ["list", "of", "entries"]}.

    Warnings
    --------
    Make sure to have `txt` files as dictionaries with explicit names. One line
    corresponds to one entry. The tagging made afterward is case insensitive.
    """
    gazetteers = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            gazetteers[file.rstrip(".txt")] = list(
                set(f.read().lower().splitlines()) # Making sure each entry is unique
            )
    return gazetteers

def escape(text: str):
    pattern = re.compile(r"([^\w+])", re.UNICODE)
    escaped_text = re.sub(pattern, r" \1 ", text)
    escaped_text = re.sub(r"\s+", " ", escaped_text)
    escaped_text = re.sub(r"\s$", "", escaped_text)
    return escaped_text

def read_conll(path: str):
    """Reads a ``conll`` file and returns a tuple containing the list of tokens
    per doc and tags epr doc.

    Parameters
    ----------
    path: ``str``
        Path to the conll file.

    Returns
    -------
    token_docs: ``list``
        List of tokens per document.
    tag_docs: ``list``
        List of labels per document.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    raw_docs = re.split(r"\n\t?\n", raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split("\n"):
            token, tag = line.split("\t")
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs

