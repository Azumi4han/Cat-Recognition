mod test;

use std::fmt::Debug;
use hdf5::{Dataset, H5Type};
use ndarray::prelude::*;
use ndarray::{Ix, RawData};
use ndarray::Data;
use ndarray::OwnedRepr;
use ndarray::DataMut;
use ndarray::linalg::Dot;
use ndarray::LinalgScalar;
use hdf5::file::File;
use hdf5::types::FixedAscii;
use crate::test::testing;


fn sigmoid<T, A>(z: &ArrayBase<T, A>) -> ArrayBase<OwnedRepr<f64>, A>
    where
        T: RawData + Data<Elem = f64>,
        A: Dimension, <T as RawData>::Elem: Clone,
{
    let arr = -z;
    let s = 1. / (1. + arr.mapv(f64::exp));
    return s;
}

fn sigmoid2(z: &Array<f64, Ix4>) -> Array<f64, Ix4> {
    let arr = -z;
    let s = 1. / (1. + arr.mapv(f64::exp));
    return s;
}

//
// fn sigmoid<T, A>(z: &ArrayBase<T, Ix2>) -> ArrayBase<OwnedRepr<f64>, Ix2>
//     where
//         T: RawData + Data<Elem = f64>,
//         A: Dimension, <T as RawData>::Elem: Clone,
// {
//     let s = 1. / (1. + z.mapv(f64::exp));
//     return s;
// }


//ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>
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
    // let b = 1.5;
    // let test = array![[1., -2., -1.], [3., 0.5, -3.2]] + b;
    // // let A = sigmoid(w.t().dot(&X) + b);
    // let bb = w.t().dot(&test);

   //let A = sigmoid(w.t().dot(&array![[1., -2., -1.], [3., 0.5, -3.2]]));
    // let A = sigmoid(w.t().dot(&X));
    let A = sigmoid(&(w.t().dot(X) + b));
    let cost = -1. / m * (Y * &A.mapv(f64::ln) + (1. - Y) * (1. - &A).mapv(f64::ln)).sum();

    let dw = 1. / m * X.dot(&(&A - Y).t());
    let db = 1. / m * (A - Y).sum();
    // let db = 1 / m *

    // let y = array![[1., 1., 0.]] * A.ln_1p();

    // let cost = -1 / m *

    // let aaa = test.t().dot(&(array![2.0] - array![1.0]));
    // println!("{:?}",  bb);
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

    //w.reshape(X.shape[0], 1)

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

// fn predict2(w: &Array2<f64>, b: f64, X: Array2<f64>) -> Array2<f64> {
//     let m = X.len_of(Axis(1));
//     let mut y_prediction =  Array2::<f64>::zeros((1, m).f());
//
//     //w.reshape(X.shape[0], 1)
//
//     let w = Array::from_shape_vec((X.len_of(Axis(0)), 1), w.into_raw_vec()).unwrap();
//
//     let A = sigmoid(&(w.t().dot(&X) + b));
//
//     for n in 0..A.len_of(Axis(1)) {
//         if A[[0, n]] > 0.5 {
//             y_prediction[[0, n]] = 1.
//         }
//         else {
//             y_prediction[[0, n]] = 0.
//         }
//     }
//
//     return y_prediction;
// }



// fn open_dataset() -> hdf5::Result<(Array1<f64>)> {
//     // Train datasets
//     let train_dataset = File::open("./dataset/train_catvnoncat.h5")?;
//     let test_dataset = File::open("./dataset/test_catvnoncat.h5")?;
//     // Train variables
//     let open_train_x = train_dataset.dataset("/train_set_x")?;
//     let open_train_y = train_dataset.dataset("/train_set_y")?;
//
//     let train_set_x = array![open_train_x.read_raw::<f64>()?];
//     let train_set_y = array![open_train_y.read_raw::<f64>()?];
//
//     let open_test_x = test_dataset.dataset("/test_set_x")?;
//     let open_test_y = test_dataset.dataset("/test_set_y")?;
//
//     let test_set_x = array![open_test_x.read_raw::<f64>()?];
//     let test_set_y = array![open_test_y.read_raw::<f64>()?];
//
//     let open_labels = test_dataset.dataset("/list_classes")?;
//
//     let label = open_labels.read_raw::<FixedAscii<10>>()?;
//
//
//     return Ok(train_set_y);
// }

fn model(x_train: Array2<f64>, y_train: Array1<f64>, x_test: Array2<f64>, y_test: Array1<f64>, num_iterations: i64, learning_rate: f64, print_cost: bool)  {
    println!("{:?}", x_test.shape());
    let zeros = initialize_with_zeros(x_train.len_of(Axis(0)));

    let optz = optimize(&zeros.0, zeros.1, &x_train, &y_train, num_iterations, learning_rate, print_cost);

    let w = optz.0;
    let b = optz.1;




    let y_prediction_test = predict(&w, b, x_test);
    let y_prediction_train = predict(&w, b, x_train);
    //
    if print_cost {
        println!("train accuracy: {:?}", 100. - (&y_prediction_train - y_train).mean().unwrap().abs() * 100.);
        println!("test accuracy: {:?}", 100. - (&y_prediction_test - y_test).mean().unwrap().abs() * 100.)
    }
    //
    //
    // return (optz.2, y_prediction_test, y_prediction_train, w.clone(), b, learning_rate, num_iterations);

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

    let label = open_labels.read_raw::<FixedAscii<10>>().unwrap();

    let num_px = train_set_x.len_of(Axis(1)) as f64;


    let train_set_x_flatten =
        Array::from_shape_vec((12288, 209), train_set_x.into_raw_vec()).unwrap();

    let test_set_x_flatten =
        Array::from_shape_vec((12288, 50), test_set_x.into_raw_vec()).unwrap();


    let train_set_x1 = train_set_x_flatten / 255.;
    let test_set_x1 = test_set_x_flatten / 255.;

    let linear_regression_model =
        model(train_set_x1, train_set_y, test_set_x1, test_set_y, 5000, 0.005, true);

    // let train_set_x_flatten =
    //     Array::from_shape_vec((train_set_x.len_of(Axis(0)), 1), train_set_x.into_raw_vec()).unwrap();
    // let test_set_x_flatten =
    //     Array::from_shape_vec((test_set_x.len_of(Axis(0)), -1.), test_set_x.into_raw_vec()).unwrap();


    //println!("{:?}", r);


    // let w = array![[1.], [2.]];
    // let b = 1.5;
    // let X = array![[1., -2., -1.], [3., 0.5, -3.2]];
    // let Y = array![[1., 1., 0.]];
    //
    // propagate(&w, b, &X, &Y);
    // let optz = optimize(&w, b, &X, &Y, 100, 0.009, true);
    //println!("{:?} HERE", optz.0);

}
