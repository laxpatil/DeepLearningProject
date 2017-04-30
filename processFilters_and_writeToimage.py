import numpy as np

from vis.utils.utils import stitch_images
from vis.visualization import visualize_activation, get_num_filters

from keras.models import Sequential, model_from_json
def load_model():
    
    run =  2
    file_name = "TYPE_gender_LR_1e-05_OPTIMIZER_rmsprop_FOLD_0"
    
    model_file = 'pretrain_on_age_and_then_on_gender/run{}/{}/model.json'.format(run, file_name)
    weight_file = 'pretrain_on_age_and_then_on_gender/run{}/{}/weights.h5'.format(run, file_name)
    
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.summary()
    loaded_model.load_weights(weight_file)
    
    print("Loaded model from disk\n ")
    return loaded_model

model = load_model()

for layer in model.layers:
    weights = layer.get_weights()
    
layer_num = 8
print (model.layers[layer_num].get_weights()[0].shape)
print model.summary()

#layer_name = 'age'
#layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

filters = np.arange(get_num_filters(model.layers[layer_num]))
# Generate input image for each filter. Here `text` field is used to overlay `filter_value` on top of the image.
#vis_images = [visualize_activation(model, layer_idx, filter_indices=idx, text=str(idx), max_iter=500)
#              for idx in [4]]
vis_images = [visualize_activation(model, layer_num, filter_indices=idx, text=str(idx))
             for idx in filters]
#print type(vis_images)
#print type(vis_images[0])

final_image = stitch_images(vis_images, cols=8)

import scipy.misc
file_name_output = 'age_layer_pred{}.jpg'.format(layer_num)
scipy.misc.imsave(file_name_output, final_image)
