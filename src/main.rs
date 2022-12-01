use std::path::{Path, PathBuf};
use ndarray::prelude::*;
use ndarray::RawData;
use ndarray::Data;
use ndarray::OwnedRepr;
use hdf5::file::File;
use hdf5::types::FixedAscii;
use image::GenericImageView;
use image::imageops::FilterType;

// Sigmoid function
fn sigmoid<T, A>(z: &ArrayBase<T, A>) -> ArrayBase<OwnedRepr<f64>, A>
    where
        T: RawData + Data<Elem = f64>,
        A: Dimension, <T as RawData>::Elem: Clone,
{
    let arr = -z;
    let s = 1. / (1. + arr.mapv(f64::exp));
    return s;
}


fn initialize_with_zeros(dim: usize) -> (Array2<f64>, f64) {

    let w = Array::zeros((dim, 1).f());
    let b = 0.0;

    return (w, b);
}

fn propagate<T2, T3>(w: &Array2<f64>, b: f64, X: &ArrayBase<T2, Ix2>, Y: &ArrayBase<T3, Ix1>)
-> (Array2<f64>, f64, f64)
where
    T2: Data<Elem = f64>,
    T3: Data<Elem = f64>,
{

    let m = X.len_of(Axis(1)) as f64;
    let A = sigmoid(&(w.t().dot(X) + b));
    let cost = -1. / m * (Y * &A.mapv(f64::ln) + (1. - Y) * (1. - &A).mapv(f64::ln)).sum();

    let dw = 1. / m * X.dot(&(&A - Y).t());
    let db = 1. / m * (A - Y).sum();

    return (dw, db, cost);
}

fn optimize<T2, T3>(w: &Array2<f64>, b: f64, X: &ArrayBase<T2, Ix2>, Y: &ArrayBase<T3, Ix1>, num_iterations: i64, learning_rate: f64, print_cost: bool)
-> (Array2<f64>, f64, Array2<f64>, f64, Vec<f64>)
where
    T2: Data<Elem = f64>,
    T3: Data<Elem = f64>,
{

    let mut w = w.clone();
    let mut b = b.clone();

    let mut costs:Vec<f64> = vec![];

    let mut prop = propagate(&w, b, &X, &Y);

    let mut dw = prop.0;
    let mut db = prop.1;
    let mut cost = prop.2;

    for n in 0..num_iterations {

        prop = propagate(&w, b, &X, &Y);
        dw = prop.0;
        db = prop.1;
        cost = prop.2;


        w = w - learning_rate * &dw;
        b = b - learning_rate * db;


        if n % 100 == 0 {
            costs.push(cost);
            if print_cost {
                println!("Cost after iteration {}: {}", n, cost);
            }
        }

    }

    return (w, b, dw, db, costs);
}

fn predict(w: &Array2<f64>, b: f64, X: Array2<f64>) -> Array2<f64> {
    let m = X.len_of(Axis(1));
    let mut y_prediction =  Array2::<f64>::zeros((1, m).f());

    let w = Array::from_shape_vec((X.len_of(Axis(0)), 1), w.clone().into_raw_vec()).unwrap();

    let A = sigmoid(&(w.t().dot(&X) + b));

    for n in 0..A.len_of(Axis(1)) {
        if A[[0, n]] > 0.5 {
            y_prediction[[0, n]] = 1.
        }
        else {
            y_prediction[[0, n]] = 0.
        }
    }

    return y_prediction;
}

fn model(x_train: Array2<f64>, y_train: Array1<f64>, x_test: Array2<f64>, y_test: Array1<f64>, num_iterations: i64, learning_rate: f64, print_cost: bool) -> (Array2<f64>, Array2<f64>, Array2<f64>, ArrayBase<OwnedRepr<f64>, Ix2>, f64, f64, i64) {

    let zeros = initialize_with_zeros(x_train.len_of(Axis(0)));


    let optz = optimize(&zeros.0, zeros.1, &x_train, &y_train, num_iterations, learning_rate, print_cost);

    let w = optz.0;
    let b = optz.1;

    let y_prediction_test = predict(&w, b, x_test);
    let y_prediction_train = predict(&w, b, x_train);

    if print_cost {
        println!("train accuracy: {:?}", 100. - (&y_prediction_train - y_train).mean().unwrap().abs() * 100.);
        println!("test accuracy: {:?}", 100. - (&y_prediction_test - y_test).mean().unwrap().abs() * 100.)
    }

    return (optz.2, y_prediction_test, y_prediction_train, w, b, learning_rate, num_iterations);

}

fn open_image(path: &str) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let img = image::open(format!("./dataset/my/{}", path)).unwrap();

    let new_dim = Dim([1, 12288]);

    let img_a = img.resize_exact(64, 64, FilterType::Nearest).as_rgb8().unwrap()
        .to_vec().iter().map(|&e| e as f64 / 255.).collect::<Vec<f64>>();

    let image = Array2::from_shape_vec(new_dim, img_a).unwrap().reversed_axes();

    return image;
}

fn main() {
    //testing(5000, 0.005, true);
    let train_dataset = File::open("./dataset/train_catvnoncat.h5").unwrap();
    let test_dataset = File::open("./dataset/test_catvnoncat.h5").unwrap();
    // Train variables
    let open_train_x = train_dataset.dataset("/train_set_x").unwrap();
    let open_train_y = train_dataset.dataset("/train_set_y").unwrap();

    let train_set_x = open_train_x.read::<f64, Ix4>().unwrap().clone();
    let train_set_y = open_train_y.read::<f64, Ix1>().unwrap().clone();

    let open_test_x = test_dataset.dataset("/test_set_x").unwrap();
    let open_test_y = test_dataset.dataset("/test_set_y").unwrap();

    let test_set_x = open_test_x.read::<f64, Ix4>().unwrap();
    let test_set_y = open_test_y.read::<f64, Ix1>().unwrap();

    let open_labels = test_dataset.dataset("/list_classes").unwrap();

    let label = open_labels.read::<FixedAscii<10>, Ix1>().unwrap();

   // let num_px = train_set_x.len_of(Axis(1));


    let train_set_x_flatten =
        Array::from_shape_vec((train_set_x.len_of(Axis(0)), 12288), train_set_x.into_raw_vec()).unwrap();
    //
    let test_set_x_flatten =
        Array::from_shape_vec((test_set_x.len_of(Axis(0)), 12288), test_set_x.into_raw_vec()).unwrap();

    let train_set_x1 = train_set_x_flatten.reversed_axes() / 255.;
    let test_set_x1 = test_set_x_flatten.reversed_axes()  / 255.;

    let logistic_regression_model =
        model(train_set_x1, train_set_y, test_set_x1, test_set_y, 7000, 0.05, true);

    let image = open_image("tiger.jpg");

    let my_predicted_image = predict(&logistic_regression_model.3, logistic_regression_model.4, image);
    println!("{:?}", label.slice(s![my_predicted_image.sum() as i32]));
}
