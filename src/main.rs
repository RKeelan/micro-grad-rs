use std::fmt::Display;

use clap::Parser;
use num_traits::Float;

use micro_grad::Layer;
use micro_grad::Neuron;
use micro_grad::MultiLayerPerceptron;
use micro_grad::Scalar;

fn tensor_test() {
    println!("\n------ MLP ------");
    let x1 = Scalar::new(2.0); x1.set_label("x1");
    let x2 = Scalar::new(0.0); x2.set_label("x2");
    let w1 = Scalar::new(-3.0); w1.set_label("w1");
    let w2 = Scalar::new(1.0); w2.set_label("w2");
    let b = Scalar::new(6.8814); b.set_label("b");
    let x1w1 = &x1 * &w1; x1w1.set_label("x1w1");
    let x2w2 = &x2 * &w2; x2w2.set_label("x2w2");
    let x1w1_x2w2 = &x1w1 + &x2w2; x1w1_x2w2.set_label("x1w1_x2w2");
    let sum = &x1w1_x2w2 + &b; sum.set_label("sum");
    let tanh = sum.tanh(); tanh.set_label("tanh");
    tanh.backward();
    println!("{}", tanh);       // tanh { data: 0.7071, grad: 1.0000 }
    println!("{}", sum);        // sum { data: 0.8814, grad: 0.5000 }
    println!("{}", x1w1_x2w2);  // x1w1_x2w2 { data: -6.0000, grad: 0.5000 }
    println!("{}", x2w2);       // x2w2 { data: 0.0000, grad: 0.5000 }
    println!("{}", x1w1);       // x1w1 { data: -6, grad: 0.5000 }
    println!("{}", w2);         // w2 { data: 1.0000, grad: 0.0000 }
    println!("{}", w1);         // w1 { data: -3.0000, grad: 1.0000 }

    println!("\n------ Variable re-use ------");
    let a = Scalar::new(3.0); a.set_label("a");
    let b = &a + &a; b.set_label("b");
    b.backward();
    println!("{}", a); // a { data: 3, grad: 2 }
    println!("{}", b); // b { data: 6, grad: 1 }

    println!("\n------ Number addition ------");
    let x = Scalar::new(2.0);
    let number = x.add_number(1.0); number.set_label("number");
    println!("{}", number); // number { data: 3, grad: 0 }

    println!("\n------ Exp ------");
    let exp = x.exp(); exp.set_label("exp");
    println!("{}", exp); // exp { data: 7.38905609893065, grad: 0 }

    println!("\n------ Power ------");
    let x = Scalar::new(3.0); x.set_label("x");
    let power = x.pow(2.0); power.set_label("power");
    power.backward();
    println!("{}", power); // power { data: 9.0, grad: 1 }
    println!("{}", x); // x { data: 3, grad: 6 }


    println!("\n------ Division ------");
    let x = Scalar::new(2.0); x.set_label("x");
    let y = Scalar::new(4.0); y.set_label("y");
    let div = &x / &y; div.set_label("div");
    div.backward();
    println!("{}", div); // div { data: 0.5, grad: 0 }
    println!("{}", x); // x { data: 0.5, grad: 0 }
    println!("{}", y); // y { data: 0.5, grad: 0 }

    println!("\n------ Subtraction ------");
    let sub = &x - &y; sub.set_label("sub");
    sub.backward();
    println!("{}", sub); // sub { data: -2.0, grad: 0 }

    println!("\n------ Manual tanh ------");
    let x1 = Scalar::new(2.0); x1.set_label("x1");
    let x2 = Scalar::new(0.0); x2.set_label("x2");
    let w1 = Scalar::new(-3.0); w1.set_label("w1");
    let w2 = Scalar::new(1.0); w2.set_label("w2");
    let b = Scalar::new(6.8814); b.set_label("b");
    let x1w1 = &x1 * &w1; x1w1.set_label("x1w1");
    let x2w2 = &x2 * &w2; x2w2.set_label("x2w2");
    let x1w1_x2w2 = &x1w1 + &x2w2; x1w1_x2w2.set_label("x1w1_x2w2");
    let sum = &x1w1_x2w2 + &b; sum.set_label("sum");
    let manual_tanh = &(sum.mul_number(2.0).exp().add_number(-1.0)) / &(sum.mul_number(2.0).exp().add_number(1.0));
    manual_tanh.set_label("manual_tanh");
    manual_tanh.backward();
    println!("{}", manual_tanh);// manual_tanh { data: 0.7071, grad: 1.0000 }
    println!("{}", sum);        // sum { data: 0.8814, grad: 0.5000 }
    println!("{}", x1w1_x2w2);  // x1w1_x2w2 { data: -6.0000, grad: 0.5000 }
    println!("{}", x2w2);       // x2w2 { data: 0.0000, grad: 0.5000 }
    println!("{}", x1w1);       // x1w1 { data: -6, grad: 0.5000 }
    println!("{}", w2);         // w2 { data: 1.0000, grad: 0.0000 }
    println!("{}", w1);         // w1 { data: -3.0000, grad: 1.0000 }
}

fn mse<T: Float + Copy + Display + std::ops::AddAssign + 'static>(ground_truths: &Vec<Scalar<T>>, predictions: &Vec<Scalar<T>>) -> Scalar<T> {
    let mut loss = Scalar::new(T::zero());
    for (gt, pred) in ground_truths.iter().zip(predictions.iter()) {
        let diff_squared = (pred - gt).pow(T::from(2.0).unwrap());
        loss = &loss + &diff_squared;
    }
    // TODO: This is what I'd use if I could get the Sum trait implemented
    //let loss: Scalar<f64> = ys.iter().zip(y_predictions.iter()).map(|(ygt, y_pred)| (y_pred - ygt).pow(2.0)).sum();
    loss
}

fn nn_test() {

    println!("\n------ Neuron ------");
    let neuron = Neuron::new(1);
    let inputs = vec![0.5];
    let prediction = neuron.forward_with_numbers(&inputs).expect(""); prediction.set_label("neuron_prediction");
    let gt =  Scalar::new_with_label(0.5, "neuron_gt");
    let loss = (&prediction - &gt).pow(2.0);
    loss.set_label("neuron_loss");
    // loss.backward();
    println!("{}", loss);
    println!("{}", neuron.weights[0]);

    println!("\n------ Layer ------");
    let layer = Layer::<f64>::new(2,1);
    let inputs = vec![0.5,-0.5];
    let prediction = &layer.forward_with_numbers(&inputs)[0]; prediction.set_label("layer_prediction");
    let gt =  Scalar::new_with_label(0.5, "layer_gt");
    let loss = (prediction - &gt).pow(2.0);
    loss.set_label("layer_loss");
    // loss.backward();
    println!("{}", loss);
    println!("{}", layer.neurons[0].weights[0]);

    println!("\n------ Simple Multi-Layer Perceptron ------");
    let mlp: MultiLayerPerceptron<f64> = MultiLayerPerceptron::new(3, vec![4, 1]);
    let inputs = vec![2.0, 3.0, -1.0];
    let prediction = &mlp.forward(&inputs)[0]; prediction.set_label("mlp_prediction");
    let gt = Scalar::new_with_label(-1.0, "mlp_gt");
    let loss = (prediction - &gt).pow(2.0);
    loss.set_label("mlp_loss");
    // loss.backward();
    println!("{}", loss);
    println!("{}", mlp.layers[0].neurons[1].weights[0]);

    println!("\n------ Complex Multi-Layer Perceptron ------");
    let mlp = MultiLayerPerceptron::new(3, vec![4,4,1]);
    let xs = [
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0].map(|y| Scalar::new(y)); // Desired output for row of xs
    // flat_map with, I think, concatenate all the outputs together. I know it's one each, and I want to combine them
    // all into one array, so this is fine, but I'd need to do something more sophisticated for the general case
    let y_predictions: Vec<Scalar<f64>> = xs.iter().flat_map(|x| mlp.forward(x)).collect();
    // for prediction in &y_predictions {
    //     println!("{}", prediction);
    // }

    let loss = mse(&ys.to_vec(), &y_predictions);
    loss.set_label("loss");
    loss.backward();
    println!("{}", loss);

    // println!("\n------ {} Parameters ------", mlp.parameters().len());
    // mlp.parameters().iter().for_each(|p| println!("{}", p));

    println!("\n------ Optimize ------");
    println!("{}", mlp.layers[0].neurons[0].weights[0]);
    let learning_rate = 0.01;
    for p in mlp.parameters() {
        p.add_to_data(p.get_grad() * -learning_rate);
    }
    println!("{}", mlp.layers[0].neurons[0].weights[0]);

    println!("\n------ Additional Forward Passes ------");
    for _ in 0..5 {
        let y_predictions: Vec<Scalar<f64>> = xs.iter().flat_map(|x| mlp.forward(x)).collect();
        let loss = mse(&ys.to_vec(), &y_predictions);
        loss.set_label("loss");
        mlp.zero_grad();
        loss.backward();
        println!("{}", loss);
        for p in mlp.parameters() {
            p.add_to_data(p.get_grad() * -learning_rate);
        }
    }
}

fn train() {
    let learning_rate = 0.1;
    let mlp = MultiLayerPerceptron::new(3, vec![4,4,1]);
    let xs = [
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0].map(|y| Scalar::new(y)); 

    for k in 0..10 {
        // Forward pass
        let y_predictions: Vec<Scalar<f64>> = xs.iter().flat_map(|x| mlp.forward(x)).collect();
        let loss = mse(&ys.to_vec(), &y_predictions);

        // Backward pass
        mlp.zero_grad();
        loss.backward();

        // Update
        for p in mlp.parameters() {
            p.add_to_data(p.get_grad() * -learning_rate);
        }
        println!("Step {} loss: {:0.4}", k, loss.get_data());
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Action to perform
    #[arg(value_parser)]
    action: String,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    match args.action.as_str() {
        "tensor" => tensor_test(),
        "nn" => nn_test(),
        "train" => train(),
        _ => {
            eprintln!("Unknown action: {}", args.action);
            std::process::exit(1);
        }
    }
}
