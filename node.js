const axios = require("axios");
const tf = require("@tensorflow/tfjs-node");
const nsfw = require("nsfwjs");

module.exports = function (RED) {
    function FunctionNode(n) {
        RED.nodes.createNode(this, n);
        var node = this;
        this.name = n.name;
        for (var key in n) {
            node[key] = n[key] || "";
        }
        // var model;
        // async function load_fn() {
        //     model = await nsfw.load() // To load a local model, nsfw.load('file://./path/to/model/')
        // }
        // load_fn();
        this.on('input', function (msg) {
            for (var i in msg) {
                if (i !== 'req' | i !== 'res' | i !== 'payload' | i !== 'send' | i !== '_msgid') {
                    node[i] = node[i] || msg[i];
                }
            }

            async function fn() {
                const model = await nsfw.load() // To load a local model, nsfw.load('file://./path/to/model/')
                const pic = await axios.get(node.url, {
                    responseType: 'arraybuffer',
                })

                // Image must be in tf.tensor3d format
                // you can convert image to tf.tensor3d with tf.node.decodeImage(Uint8Array,channels)
                const image = await tf.node.decodeImage(pic.data, 3)
                const predictions = await model.classify(image)
                image.dispose() // Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
                node.error(predictions);
                msg.payload = predictions;
                node.send(msg);
            }
            fn();
        });
    }

    RED.nodes.registerType("nsfw", FunctionNode);
};
