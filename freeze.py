import os
import freeze_graph
import tensorflow as tf

checkpoint_state_name = "model.ckpt"
input_graph_name = "starnet.pb"
output_graph_name = "starne_weights.pb"

input_graph_path = os.path.join("./", input_graph_name)
input_saver_def_path = ""
input_binary = False
input_checkpoint_path = os.path.join("./", 'saved_checkpoint')

print(tf.version.VERSION)

output_node_names = "generator/g_deconv7/Sub"
restore_op_name = "save/restore_all" # not used
filename_tensor_name = "save/Const:0" # not used
output_graph_path = os.path.join("./", output_graph_name)
clear_devices = False

freeze_graph.freeze_graph("./starnet_generator.pb",
                          "",
                          True,
                          "./model.ckpt",
                          output_node_names,
                          restore_op_name,
                          filename_tensor_name,
                          "./starnet_generator_weights.pb",
                          clear_devices,
                          "")