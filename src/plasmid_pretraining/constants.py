"""Constants and configuration for plasmid data processing."""

# Data sources
TAR_FILES = [
    "COMPASS.gbk.tar.gz",
    "DDBJ.gbk.tar.gz",
    "ENA.gbk.tar.gz",
    "GenBank.gbk.tar.gz",
    "IMG-PR.gbk.tar.gz",
    "Kraken2.gbk.tar.gz",
    "PLSDB.gbk.tar.gz",
    "RefSeq.gbk.tar.gz",
    "TPA.gbk.tar.gz",
    "mMGE.gbk.tar.gz",
]

# Resistance markers and their patterns
RESISTANCE_MARKERS = {
    "ampicillin": ["ampicillin", "bla", "amp", "beta-lactam", "penicillin"],
    "kanamycin": ["kanamycin", "kan", "aph", "neomycin", "neo"],
    "chloramphenicol": ["chloramphenicol", "cat", "cm"],
    "tetracycline": ["tetracycline", "tet"],
    "spectinomycin": ["spectinomycin", "spec", "aad"],
    "streptomycin": ["streptomycin", "str", "aada"],
    "gentamicin": ["gentamicin", "gent", "aac"],
    "hygromycin": ["hygromycin", "hyg", "hph"],
    "puromycin": ["puromycin", "puro", "pac"],
    "blasticidin": ["blasticidin", "bsd", "bsr"],
    "zeocin": ["zeocin", "ble", "sh ble"],
    "erythromycin": ["erythromycin", "erm"],
    "trimethoprim": ["trimethoprim", "dfr", "dhfr"],
}

# Reporter genes
REPORTER_GENES = {
    "EGFP": ["egfp", "gfp", "green fluorescent"],
    "mCherry": ["mcherry", "cherry"],
    "mRFP": ["mrfp", "rfp", "red fluorescent"],
    "BFP": ["bfp", "blue fluorescent", "ebfp"],
    "YFP": ["yfp", "yellow fluorescent", "eyfp", "venus"],
    "CFP": ["cfp", "cyan fluorescent", "ecfp"],
    "tdTomato": ["tdtomato", "tomato"],
    "mVenus": ["mvenus"],
    "mCitrine": ["mcitrine", "citrine"],
    "luciferase": ["luciferase", "luc", "nanoluc", "fluc", "gluc"],
    "lacZ": ["lacz", "beta-galactosidase", "b-gal"],
}

# Protein tags
PROTEIN_TAGS = {
    "his": ["his-tag", "his6", "6xhis", "histidine tag", "polyhistidine"],
    "flag": ["flag", "dykddddk"],
    "myc": ["myc", "c-myc"],
    "ha": ["ha-tag", "hemagglutinin"],
    "gst": ["gst", "glutathione"],
    "mbp": ["mbp", "maltose binding"],
    "strep": ["strep-tag", "streptavidin"],
    "v5": ["v5 tag", "v5-tag"],
    "sumo": ["sumo"],
    "t7": ["t7 tag"],
}

# Plasmid type keywords
PLASMID_TYPES = {
    "mammalian_expression": ["mammalian", "hek", "cho", "hela", "cos", "cmv", "sv40", "ef1a", "cag", "pgk"],
    "bacterial_expression": ["bacterial", "e. coli", "ecoli", "t7", "plac", "ptac", "para", "ptrc"],
    "yeast_expression": ["yeast", "saccharomyces", "pichia", "gal1", "gal10", "adh1"],
    "insect_expression": ["insect", "baculovirus", "sf9", "sf21", "hi5"],
    "plant_expression": ["plant", "agrobacterium", "35s", "camv"],
    "lentiviral": ["lentivirus", "lentiviral", "ltr", "psi packaging"],
    "retroviral": ["retrovirus", "retroviral", "mmlv", "moloney"],
    "adenoviral": ["adenovirus", "adenoviral"],
    "aav": ["aav", "adeno-associated"],
    "crispr": ["crispr", "cas9", "cas12", "grna", "sgrna"],
    "cloning": ["cloning", "entry", "gateway", "topo", "ta cloning"],
    "shuttle": ["shuttle"],
}

# Copy number markers
COPY_NUMBER_MARKERS = {
    "high": ["pbr322", "puc", "colei", "pmb1", "high copy"],
    "medium": ["p15a", "medium copy"],
    "low": ["psc101", "f plasmid", "low copy", "single copy"],
}

# Backbone component keywords
BACKBONE_KEYWORDS = {
    "origin": ["ori", "origin", "replication", "rep", "cole1", "colei", "pbr322", "puc", "p15a", "psc101", "f1", "sv40 ori"],
    "selection": ["resistance", "bla", "kan", "cat", "tet", "amp", "neo", "hygro", "puro", "blast", "zeo", "spec", "gent", "erm", "aph", "aad"],
    "backbone_promoter": ["t7 promoter", "lac promoter", "tac promoter", "trc promoter", "ara promoter", "tet promoter"],
    "terminator": ["terminator", "term", "t7 term", "rrnb", "poly(a)", "polya", "sv40 poly", "bgh poly"],
    "plasmid_element": ["mobilization", "mob", "tra", "transfer", "conjugation", "partition", "par", "stability"],
    "viral_backbone": ["ltr", "psi", "packaging", "wpre", "itr", "inverted terminal"],
    "cloning": ["multiple cloning", "mcs", "polylinker", "cloning site"],
}

# Insert keywords
INSERT_KEYWORDS = {
    "fluorescent": ["gfp", "egfp", "mcherry", "rfp", "mrfp", "bfp", "yfp", "cfp", "venus", "citrine", "tomato", "tdtomato"],
    "reporter": ["luciferase", "luc", "fluc", "rluc", "nluc", "lacz", "beta-galactosidase", "seap"],
    "therapeutic": ["interleukin", "il-", "interferon", "ifn", "tnf", "growth factor", "cytokine", "antibody", "scfv"],
    "editing": ["grna", "sgrna", "guide rna", "target sequence", "homology arm", "donor"],
    "recombinant": ["recombinant", "fusion protein", "chimeric", "heterologous", "codon optimized"],
    "expression_cassette": ["expression cassette", "gene of interest", "goi", "transgene", "insert"],
}

# Expression promoters
EXPRESSION_PROMOTERS = {
    "mammalian": ["cmv", "ef1a", "ef-1", "cag", "pgk", "ubiquitin", "ubc", "sv40 promoter", "rsv"],
    "bacterial": ["t7", "lac", "tac", "trc", "ara", "tet", "rhamnose"],
    "yeast": ["gal1", "gal10", "adh1", "tef1", "pgk1 promoter"],
    "insect": ["polh", "polyhedrin", "p10", "ie1"],
}

# Token prefixes for training pairs
TOKEN_PREFIX = {
    "type": "TYPE",
    "resistance": "RES",
    "copy_number": "COPY",
    "insert": "INSERT",
    "tag": "TAG",
    "host": "HOST",
    "topology": "TOPO",
    "length": "LEN",
}

# Standard values for normalization
STANDARD_VALUES = {
    "type": [
        "mammalian_expression", "bacterial_expression", "yeast_expression",
        "insect_expression", "plant_expression", "lentiviral", "retroviral",
        "adenoviral", "aav", "crispr", "cloning", "shuttle", "unknown"
    ],
    "resistance": [
        "ampicillin", "kanamycin", "chloramphenicol", "tetracycline",
        "spectinomycin", "streptomycin", "gentamicin", "hygromycin",
        "puromycin", "blasticidin", "zeocin", "erythromycin", "trimethoprim",
        "multiple", "unknown"
    ],
    "copy_number": ["high", "medium", "low", "unknown"],
    "tag": ["his", "flag", "myc", "ha", "gst", "mbp", "strep", "v5", "sumo", "t7", "gfp_tag", "fc", "multiple", "none"],
    "host": ["e_coli", "mammalian", "yeast", "insect", "plant", "bacterial", "multiple", "unknown"],
    "topology": ["circular", "linear"],
}

# Length bins
LENGTH_BINS = [
    (0, 3000, "small"),
    (3000, 6000, "medium"),
    (6000, 10000, "large"),
    (10000, 20000, "very_large"),
    (20000, float("inf"), "mega"),
]

# Insert gene categories
INSERT_CATEGORIES = {
    "fluorescent": ["EGFP", "mCherry", "mRFP", "BFP", "YFP", "CFP", "tdTomato", "mVenus", "mCitrine"],
    "reporter": ["luciferase", "lacZ", "SEAP", "CAT"],
    "therapeutic": ["cytokine", "antibody", "enzyme", "hormone", "growth_factor"],
    "editing": ["gRNA", "Cas9", "donor_template"],
    "generic": ["GOI", "insert", "transgene"],
}
