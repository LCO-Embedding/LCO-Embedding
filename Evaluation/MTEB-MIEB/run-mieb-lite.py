import mteb
tasks=mteb.get_tasks(
    tasks=[
    # # Image Classification
    "Country211",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "OxfordPets",
    "PatchCamelyon",
    "RESISC45",
    "SUN397",
    
    # Clustering
    "ImageNetDog15Clustering",
    "TinyImageNetClustering",

    # ZeroShotClassification
    "CIFAR100ZeroShot",
    "Country211ZeroShot",
    "FER2013ZeroShot",
    "FGVCAircraftZeroShot",
    "Food101ZeroShot",
    "OxfordPetsZeroShot",
    "StanfordCarsZeroShot",

    # # Any2AnyMultipleChoice
    "BLINKIT2IMultiChoice",
    "CVBenchCount",
    "CVBenchRelation",
    "CVBenchDepth",
    "CVBenchDistance",

    # # ImageTextPairClassification
    "AROCocoOrder",
    "AROFlickrOrder",
    "AROVisualAttribution",
    "AROVisualRelation",
    "Winoground",
    "ImageCoDeT2IMultiChoice",

    # # Any2AnyRetrieval
    "CIRRIT2IRetrieval",
    "CUB200I2IRetrieval",
    "Fashion200kI2TRetrieval",
    "HatefulMemesI2TRetrieval",
    "InfoSeekIT2TRetrieval",
    "NIGHTSI2IRetrieval",
    "OVENIT2TRetrieval",
    "RP2kI2IRetrieval",
    "VisualNewsI2TRetrieval",
    "VQA2IT2TRetrieval",
    "WebQAT2ITRetrieval",

    # multilingual image retrieval
    "WITT2IRetrieval",
    "XM3600T2IRetrieval"

    # Doc
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTabfquadRetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreShiftProjectRetrieval",
    "VidoreSyntheticDocQAAIRetrieval",

    # Visual STS
    "STS13VisualSTS",
    "STS15VisualSTS"

    # Visual STS X&multi
    "STSBenchmarkMultilingualVisualSTS",
    "STS17MultilingualVisualSTS"
    ]
)

for model_name in [
    # "gme_7b"
    # "mme5"
    "LCO-Embedding-Omni-7B"
]:

    model = mteb.get_model(model_name)
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="./mieb-results", batch_size=8)