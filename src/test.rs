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


fn sigmoid<T, A>(z: &ArrayBase<T, A>) -> ArrayBase<OwnedRepr<f64>, A>
    where
        T: RawData + Data<Elem = f64>,
        A: Dimension, <T as RawData>::Elem: Clone,
{
    let arr = -z;
    let s = 1. / (1. + arr.mapv(f64::exp));
    return s;
}


pub fn testing(num_iterations: i64, learning_rate: f64, print_cost: bool) {
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


    // ZEROS
    let mut w_z = Array2::<f64>::zeros((train_set_x1.len_of(Axis(0)), 1).f());
    let mut b_z = 0.0;
    // ZEROS

    //PROPAGATE
    let m = train_set_x1.len_of(Axis(1)) as f64;

    let A = sigmoid(&(w_z.t().dot(&train_set_x1) + b_z));
    let cost = -1. / m * (&train_set_y * &A.mapv(f64::ln) + (1. - &train_set_y) * (1. - &A).mapv(f64::ln)).sum();

    let dw = 1. / m * train_set_x1.dot(&(&A - &train_set_y).t());
    let db = 1. / m * (A - &train_set_y).sum();
    //PROPAGATE

    //OPTIMIZE
    let mut w = w_z;
    let mut b = b_z;

    let mut costs:Vec<f64> = vec![];

    for n in 0..num_iterations {

        w = w - learning_rate * &dw;
        b = b - learning_rate * db;


        if n % 100 == 0 {
            costs.push(cost);
            if print_cost {
                println!("Cost after iteration {}: {}", n, cost);
            }
        }

    }
    //OPTIMIZE

    //PREDICTION
    let m = test_set_x1.len_of(Axis(1));
    let mut y_prediction_1 =  Array2::<f64>::zeros((1, m).f());

    //w.reshape(X.shape[0], 1)

    let w = Array::from_shape_vec((test_set_x1.len_of(Axis(0)), 1), w.into_raw_vec()).unwrap();

    let A = sigmoid(&(w.t().dot(&test_set_x1) + b));

    for n in 0..A.len_of(Axis(1)) {
        if A[[0, n]] > 0.5 {
            y_prediction_1[[0, n]] = 1.
        }
        else {
            y_prediction_1[[0, n]] = 0.
        }
    }
    //PREDICTION
    //PREDICTION
    let m = train_set_x1.len_of(Axis(1));
    let mut y_prediction_2 =  Array2::<f64>::zeros((1, m).f());

    //w.reshape(X.shape[0], 1)

    let w = Array::from_shape_vec((train_set_x1.len_of(Axis(0)), 1), w.into_raw_vec()).unwrap();

    let A = sigmoid(&(w.t().dot(&train_set_x1) + b));

    for n in 0..A.len_of(Axis(1)) {
        if A[[0, n]] > 0.5 {
            y_prediction_2[[0, n]] = 1.
        }
        else {
            y_prediction_2[[0, n]] = 0.
        }
    }
    //PREDICTION

    if print_cost {
        println!("train accuracy: {:?}", 100. - (y_prediction_2 - train_set_y).mean().unwrap().abs() * 100.);
        println!("test accuravy: {:?}", 100. - (y_prediction_1 - test_set_y).mean().unwrap().abs() * 100.)
    }
}