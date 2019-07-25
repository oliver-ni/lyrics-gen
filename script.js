let char_to_idx;
let idx_to_char;
let model;
let done_setup = false;

function choose(choices) {
    var index = Math.floor(Math.random() * choices.length);
    return choices[index];
}

let text_generated = "";
let start_char = choose("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

async function setup() {
    char_to_idx = await $.getJSON("char_to_idx.json");
    idx_to_char = await $.getJSON("idx_to_char.json");
    model = await tf.loadLayersModel("model/model.json");
    done_setup = true;

    $("#generateButton").attr("disabled", false);
    $("#output").html("Ready.")
}

async function continue_generating(num_generate) {
    $("#generateButton").attr("disabled", true);
    $("#continueButton").attr("disabled", true);

    let input_eval = char_to_idx[start_char];
    input_eval = tf.expandDims(input_eval, 0);

    for (let i = 0; i < num_generate; i++) {
        let predictions = model.predict(input_eval);
        predictions = tf.squeeze(predictions, 0);

        const predicted = await tf.multinomial(predictions, 1).array();

        const predicted_id = predicted[predicted.length - 1][0];

        input_eval = tf.expandDims([predicted_id], 0);
        text_generated += idx_to_char[predicted_id];
        start_char = idx_to_char[predicted_id];

        $("#output").html(text_generated.replace(/\n/g, "<br>"));
    }

    $("#generateButton").attr("disabled", false);
    $("#continueButton").attr("disabled", false);
}

async function generate_text(num_generate) {
    start_char = choose("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    text_generated = start_char;

    model.resetStates();

    continue_generating(num_generate);
}

setup()
