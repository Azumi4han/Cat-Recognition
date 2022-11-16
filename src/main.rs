use ndarray::prelude::*;
use ndarray::RawData;
use ndarray::Data;
use ndarray::OwnedRepr;
use ndarray::DataMut;
use ndarray::linalg::Dot;
use ndarray::LinalgScalar;


fn sigmoid<T, A>(z: &ArrayBase<T, A>) -> ArrayBase<OwnedRepr<f64>, A>
    where
        T: RawData + Data<Elem = f64>,
        A: Dimension, <T as RawData>::Elem: Clone,
{
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

    let w = Array2::<f64>::zeros((dim, 1).f());
    let b = 0.0;

    return (w, b);
}

fn propagate<T2, T3>(w: &Array2<f64>, b: f64, X: &ArrayBase<T2, Ix2>, Y: &ArrayBase<T3, Ix2>)
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

fn optimize<T2, T3>(w: &Array2<f64>, b: f64, X: &ArrayBase<T2, Ix2>, Y: &ArrayBase<T3, Ix2>, num_iterations: i64, learning_rate: f64, print_cost: bool)
where
    T2: Data<Elem = f64>,
    T3: Data<Elem = f64>,
{

    let mut w = w.clone();
    let mut b = b.clone();

    let mut costs:Vec<f64> = vec![];

    for n in 0..2 {
        let prop = propagate(&w, b, &X, &Y);

        let dw = prop.0;
        let db = prop.1;
        let cost = prop.2;

        w = w - learning_rate * dw;
        b = b - learning_rate * db;

        if n % 100 == 0 {
            println!("{}", cost);
            costs.push(cost);
        }
    }
}


fn main() {
    let a = array!([0.5, 0.0, 2.0]);
    // let sigm = sigmoid(&-a);

    // println!("{}", sigm);
   // println!("{:?}", initialize_with_zeros(2));

    let w = array![[1.], [2.]];
    let b = 1.5;
    let X = array![[1., -2., -1.], [3., 0.5, -3.2]];
    let Y = array![[1., 1., 0.]];

    propagate(&w, b, &X, &Y);
    optimize(&w, b, &X, &Y, 100, 0.009, false);

}