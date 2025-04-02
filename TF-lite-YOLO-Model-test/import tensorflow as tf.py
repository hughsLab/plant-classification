import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Path to your frozen graph
frozen_graph_path = "C:/Users/USER/Downloads/saved_model.pb"
saved_model_dir = "C:/Users/USER/Downloads/saved_model"

# Load the frozen graph
with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Export as SavedModel
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    with tf.Session(graph=graph) as sess:
        tf.saved_model.simple_save(
            sess,
            saved_model_dir,
            inputs={"input": graph.get_tensor_by_name("input_tensor_name:0")},  # Update input tensor name
            outputs={"output": graph.get_tensor_by_name("output_tensor_name:0")}  # Update output tensor name
        )
