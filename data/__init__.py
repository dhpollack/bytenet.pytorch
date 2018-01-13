from data.wmt_loader import WMT
from data.iwslt_loader import IWSLT
from data.taboeta_loader import TABOETA

def select_dataset(model_name, config):
    if model_name == "bytenet_iwslt":
        ds = IWSLT(config["IWSLT_DIR"])
    elif model_name == "bytenet_taboeta":
        ds = TABOETA(config["TABOETA_DIR"])
    else:
        if model_name != "bytenet_wmt":
            print("{} not found, defaulting to WMT".format(model_name))
        ds = WMT(config["WMT_DIR"])
    return ds
