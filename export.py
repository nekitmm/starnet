import tensorflow as tf
import model

WINDOW_SIZE = 256

def export():
    X = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "X")
    Y = tf.placeholder(tf.float32, shape = [None, WINDOW_SIZE, WINDOW_SIZE, 3], name = "Y")
    train, avers, outputs = model.model(X, Y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # initialize all variables and start training
        sess.run(init)
        # restore state of the model
        print("Restoring current state of the model...", end = ' ', flush = True)
        saver.restore(sess, "./model.ckpt")
        print("Done!", flush = True)
        print("Exporting the graph...", end = ' ', flush = True)
        tf.train.write_graph(sess.graph, './', 'starnet.pb', as_text = False)
        tf.train.write_graph(sess.graph, './', 'starnet.pbtxt', as_text = True)
        
        gen_layers = []
        
        with open("gen_sub.txt", "r") as f:
            for l in f:
                gen_layers.append(l[:-1])
        
        subgraph = tf.graph_util.extract_sub_graph(sess.graph.as_graph_def(), gen_layers)
        tf.train.write_graph(subgraph, './', 'starnet_generator.pbtxt', as_text = True)
        tf.train.write_graph(subgraph, './', 'starnet_generator.pb', as_text = False)
        print("Done!", flush = True)