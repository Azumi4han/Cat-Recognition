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
use hdf5::types::DynValue::String;
use hdf5::types::FixedAscii;
use image::{GenericImageView, ImageBuffer, Rgb};
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use ndarray_image::{NdImage, NdColor, open_image as op, Colors};
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

        println!("{:?} THIS", A[[0, n]]);
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

fn model(x_train: Array2<f64>, y_train: Array1<f64>, x_test: Array2<f64>, y_test: Array1<f64>, num_iterations: i64, learning_rate: f64, print_cost: bool) -> (Array2<f64>, Array2<f64>, Array2<f64>, ArrayBase<OwnedRepr<f64>, Ix2>, f64, f64, i64) {

    let zeros = initialize_with_zeros(x_train.len_of(Axis(0)));


    let optz = optimize(&zeros.0, zeros.1, &x_train, &y_train, num_iterations, learning_rate, print_cost);

    let w = optz.0;
    let b = optz.1;

    let y_prediction_test = predict(&w, b, x_test);
    let y_prediction_train = predict(&w, b, x_train);
    // //
    if print_cost {
        println!("train accuracy: {:?}", 100. - (&y_prediction_train - y_train).mean().unwrap().abs() * 100.);
        println!("test accuracy: {:?}", 100. - (&y_prediction_test - y_test).mean().unwrap().abs() * 100.)
    }
    //
    //

    return (optz.2, y_prediction_test, y_prediction_train, w, b, learning_rate, num_iterations);

}
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;

// fn test() -> image::error::ImageResult<()> {
//     let img = image::open("./dataset/my/cat.jpg").unwrap();
//     let (width, height) = img.dimensions();
//
//     let n = ( * height) as usize;
//
//     let new_dim = Dim([, 3]);
//
//     let img_a = img.resize_exact(64, 64, FilterType::Nearest).as_rgb8().unwrap()
//         .to_vec().iter().map(|&e| e as f64 / 255.0).collect::<Vec<f64>>();
//
//     let a = Array2::from_shape_vec(new_dim, img_a).unwrap();
//     println!("{:?}", a);
//
//     // let img_a = img.resize(64, 64, FilterType::Nearest).as_rgb8().unwrap()
//     //     .to_vec().iter().map(|&e| e as f64 / 255.0).collect::<Vec<f64>>();
// // Normalize image pixels to [0, 1]
// //     let tmp = img_a.to_vec().iter().map(|&e| e as f64 / 255.0).collect::<Vec<f64>>();
// // Reduce dimensions
//
// // Change the array values by using some other method
//
// // Image buffer for the new image
// //     let mut img_buf = image::ImageBuffer::new(width, height);
//     Ok(())
//
// }

fn open_image(num_pixels: u32) -> ArrayBase<OwnedRepr<f64>, Ix2> {
    let img = image::open("./dataset/my/cat.jpg").unwrap();
    let (width, height) = img.dimensions();

    let n = (num_pixels * num_pixels) as usize;

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

    let label = open_labels.read_raw::<FixedAscii<10>>().unwrap();

    let num_px = train_set_x.len_of(Axis(1));


    let train_set_x_flatten =
        Array::from_shape_vec((train_set_x.len_of(Axis(0)), 12288), train_set_x.into_raw_vec()).unwrap();
    //
    let test_set_x_flatten =
        Array::from_shape_vec((test_set_x.len_of(Axis(0)), 12288), test_set_x.into_raw_vec()).unwrap();

    // let a = Array::from_shape_vec((1, train_set_y.len_of(Axis(0))), train_set_y.into_raw_vec()).unwrap();

    // let testing =
    //     Array::from_shape_vec((test_set_x.len_of(Axis(0)), 0), test_set_x.into_raw_vec()).unwrap();



    let train_set_x1 = train_set_x_flatten.reversed_axes() / 255.;
    let test_set_x1 = test_set_x_flatten.reversed_axes()  / 255.;
    // let r = array![[1.0]];
    // println!("{:?}", r.len_of(Axis(0)));


    let logistic_regression_model =
        model(train_set_x1, train_set_y, test_set_x1, test_set_y, 5000, 0.005, true);

    let image = open_image(num_px as u32);


    let my_predicted_image = predict(&logistic_regression_model.3, logistic_regression_model.4, image);
    // println!("Your algorithm predicts a {:?}", label[my_predicted_image.len_of(Axis(0))])
    println!("{:?}", my_predicted_image);
    // let train_set_x_flatten =
    //     Array::from_shape_vec((train_set_x.len_of(Axis(0)), 1), train_set_x.into_raw_vec()).unwrap();
    // let test_set_x_flatten =
    //     Array::from_shape_vec((test_set_x.len_of(Axis(0)), -1.), test_set_x.into_raw_vec()).unwrap();


    //println!("{:?}", r);

    // let w = array![[0.1124579], [0.23106775]];
    // let b = -0.3;
    // let X = array![[1., -1.1, -3.2],[1.2, 2., 0.1]];
    //
    // println!("{:?}", predict(&w, b, X));

    // let w = array![[1.], [2.]];
    // let b = 1.5;
    // let X = array![[1., -2., -1.], [3., 0.5, -3.2]];
    // let Y = array![[1., 1., 0.]];
    // //
    // optimize2(&w, b, &X, &Y, 100, 0.009, false);
    // let a = propagate2(&w, b, &X, &Y);
    // println!("{:?}", a.0);
    // let optz = optimize(&w, b, &X, &Y, 100, 0.009, true);
    //println!("{:?} HERE", optz.0);

}
