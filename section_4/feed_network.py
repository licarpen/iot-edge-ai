import argparse
### TO CHECK: Load the necessary libraries
from openvino.inference_engine import IECore, IENetwork

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args


def load_to_IE(model_xml):
    ### TO CHECK: Load the Inference Engine API
    ie = IECore()
    
    ### TO CHECK: Load IR files into their related class
    net = IENetwork(model = model_xml, weights = model_xml.split('.')[0] + ".bin")
    

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    ie.add_extension(CPU_EXTENSION, 'CPU')

    ### TO CHECK: Get the supported layers of the network
    layers_supported = ie.query_network(net, 'CPU')
    
    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    layers_unsupported = [layer for layer in net.layers.keys() if layer not in layers_supported]
    if len(layers_unsupported) != 0:
        print("Unsupported layers include: {}.  Add extensions to IECore if available.".format(layers_unsupported))
        exit(1)
    
    ### TODO: Load the network into the Inference Engine
    
    ie.load_network(net, 'CPU')

    print("IR successfully loaded into Inference Engine.")

    return


def main():
    args = get_args()
    load_to_IE(args.m)


if __name__ == "__main__":
    main()
