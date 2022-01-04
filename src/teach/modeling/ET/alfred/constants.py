import os

########################################################################################################################
# General Settings

ET_ROOT = os.environ["ET_ROOT"]
ET_DATA = os.environ["ET_DATA"] if "ET_DATA" in os.environ else None
ET_LOGS = os.environ["ET_LOGS"] if "ET_LOGS" in os.environ else None

PAD = 0

########################################################################################################################

# TRAIN AND EVAL SETTINGS
# evaluation on multiple GPUs
NUM_EVAL_WORKERS_PER_GPU = 3
# vocabulary file name
VOCAB_FILENAME = "data.vocab"
# vocabulary with object classes
OBJ_CLS_VOCAB = "files/obj_cls.vocab"

#############################

OBJECTS_ACTIONS = [
    "None",
    "AlarmClock",
    "Apple",
    "AppleSliced",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Book",
    "Bowl",
    "Box",
    "Bread",
    "BreadSliced",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Candle",
    "Cart",
    "CellPhone",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Desk",
    "DeskLamp",
    "DiningTable",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "Faucet",
    "FloorLamp",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "Lettuce",
    "LettuceSliced",
    "Microwave",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Pan",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Pot",
    "Potato",
    "PotatoSliced",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "Shelf",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "TVStand",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "Tomato",
    "TomatoSliced",
    "Vase",
    "Watch",
    "WateringCan",
    "WineBottle",
]
