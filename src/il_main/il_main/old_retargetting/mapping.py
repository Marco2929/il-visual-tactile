from dex_retargeting.retargeting_config import RetargetingConfig
import numpy as np

def retarget_glove_to_hand(data_glove_joints):
    config_path = "/ros2_ws/src/il_main/il/retarget_correct.yml"
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    return retargeting.retarget(data_glove_joints)


