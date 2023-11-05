use micro_grad::Scalar;

fn main() {
    env_logger::init();

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
    tanh.back_prop();
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
    b.back_prop();
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
    power.back_prop();
    println!("{}", power); // power { data: 9.0, grad: 1 }
    println!("{}", x); // x { data: 3, grad: 6 }


    println!("\n------ Division ------");
    let x = Scalar::new(2.0); x.set_label("x");
    let y = Scalar::new(4.0); y.set_label("y");
    let div = &x / &y; div.set_label("div");
    div.back_prop();
    println!("{}", div); // div { data: 0.5, grad: 0 }
    println!("{}", x); // x { data: 0.5, grad: 0 }
    println!("{}", y); // y { data: 0.5, grad: 0 }

    println!("\n------ Subtraction ------");
    let sub = &x - &y; sub.set_label("sub");
    sub.back_prop();
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
    manual_tanh.back_prop();
    println!("{}", manual_tanh);// manual_tanh { data: 0.7071, grad: 1.0000 }
    println!("{}", sum);        // sum { data: 0.8814, grad: 0.5000 }
    println!("{}", x1w1_x2w2);  // x1w1_x2w2 { data: -6.0000, grad: 0.5000 }
    println!("{}", x2w2);       // x2w2 { data: 0.0000, grad: 0.5000 }
    println!("{}", x1w1);       // x1w1 { data: -6, grad: 0.5000 }
    println!("{}", w2);         // w2 { data: 1.0000, grad: 0.0000 }
    println!("{}", w1);         // w1 { data: -3.0000, grad: 1.0000 }

}
