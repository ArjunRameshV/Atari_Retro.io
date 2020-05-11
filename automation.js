"use strict";

//Handling the MATRIX involved   ---------------- //

class Matrix {
    constructor(row, col, data = []) {
        this._row = row;
        this._col = col;
        this._data = data;

        //initialising the matrix 
        if (data == null || data.length == 0) {
            this._data = [];
            for (let i = 0; i < row; i++) {
                this._data[i] = [];
                for (let j = 0; j < col; j++) {
                    this._data[i][j] = 0;
                }
            }
        } else { //checking whether the information enterd is correct
            if (data.length != row || data[0].length != col) {
                throw new Error("Wrong series of input to the matrix");
            }

        }

    }

    get row() {
        return this._row;
    }
    get col() {
        return this._col;
    }
    get data() {
        return this._data;
    }

    //adding two matrices
    static add(m0, m1) {
        if (!Matrix.checkDimensions(m0, m1)) {
            throw new Error("The dimensions for addition are wrong!");
        }
        var m = new Matrix(m0.row, m0.col);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                m.data[i][j] = m0.data[i][j] + m1.data[i][j];
            }
        }
        return m;
    }

    //subtract two matrices
    static subtract(m0, m1) {
        if (!Matrix.checkDimensions(m0, m1)) {
            throw new Error("The dimensions for addition are wrong!");
        }
        var m = new Matrix(m0.row, m0.col);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                m.data[i][j] = m0.data[i][j] - m1.data[i][j];
            }
        }
        return m;
    }

    //(normal) multiplication two matrices
    static multiply(m0, m1) {
        if (!Matrix.checkDimensions(m0, m1)) {
            throw new Error("The dimensions for addition are wrong!");
        }
        var m = new Matrix(m0.row, m0.col);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                m.data[i][j] = m0.data[i][j] * m1.data[i][j];
            }
        }
        return m;
    }

    //dot product for two matrices
    static dot(m0, m1) {
        if (m0.col != m1.row) {
            throw new Error("The dimension for \"dot\" product are wrong!")
        }
        var m = new Matrix(m0.row, m1.col);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                var sum = 0;
                for (let k = 0; k < m0.col; k++) {
                    sum += m0.data[i][k] * m1.data[k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }

    //converting an array to a matrix
    static convertArrayToMatrix(arr) {
        return new Matrix(1, arr.length, [arr]);
    }

    //applying a function to all the elements of a matrix
    static map(m0, func) {
        var m = new Matrix(m0.row, m0.col);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                m.data[i][j] = func(m0.data[i][j]);
            }
        }
        return m;
    }

    //finding the transpose of a matrix
    static transpose(m0) {
        var m = new Matrix(m0.col, m0.row);
        for (let i = 0; i < m.row; i++) {
            for (let j = 0; j < m.col; j++) {
                m.data[i][j] = m0.data[j][i];
            }
        }
        return m;
    }

    //checking the dimensions for two given matrices
    static checkDimensions(m0, m1) {
        if (m0.row != m1.row || m0.col != m1.col)
            return false;
        return true;
    }

    //assigning random weights between -1 and 1
    randomWeigths() {
        for (let i = 0; i < this.row; i++) {
            for (let j = 0; j < this.col; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

}

// ------------------------------------------------- End of the Matrix 

//--------          The Neural Network         -----------/

class neuralNetwork {
    constructor(ipNum, hNum, opNum) {
        this._ipNum = ipNum;
        this._hNum = hNum;
        this._opNum = opNum;
        this._weights0 = new Matrix(ipNum, hNum);
        this._weights1 = new Matrix(hNum, opNum);
        this._hidden;
        this._input;

        // generating randomized weights initially
        this._weights0.randomWeigths();
        this._weights1.randomWeigths();
        // console.table(this._weights1.data);
    }

    // getters and setters
    get weights0() {
        return this._weights0;
    }
    set weights0(w) {
        this._weights0 = w;
    }

    get weights1() {
        return this._weights1;
    }
    set weights1(w) {
        this._weights1 = w;
    }

    get hidden() {
        return this._hidden;
    }
    set hidden(h) {
        this._hidden = h;
    }

    get input() {
        return this._input;
    }
    set input(i) {
        this._input = i;
    }

    //the sketch for the motion of data through the neural network 
    feedForward(ipArr) {
        // the input matrix 
        this.input = Matrix.convertArrayToMatrix(ipArr);

        // the hidden layers matrix
        this.hidden = Matrix.dot(this.input, this.weights0);
        this.hidden = Matrix.map(this.hidden, x => sigmoid(x)); //activation on hidden function

        //the output layers matrix
        let output = Matrix.dot(this.hidden, this.weights1);
        output = Matrix.map(output, x => sigmoid(x));

        return output;

        //the bias TODO;
    }

    //the training happens here with inputArray and targetArray !!
    train(ipArr, tarArr) {

        let output = this.feedForward(ipArr); //we obtain the output for out input array
        let target = this.feedForward(tarArr); //we obtain the target array
        let opError = Matrix.subtract(target, output);
        let opDeri = Matrix.map(output, x => sigmoid(x, true)); //finding the derivative of output matrix
        let opDelta = Matrix.multiply(opDeri, opError); //finding the output delta / deviation in output layer

        let w1Trans = Matrix.transpose(this.weights1);
        let hError = Matrix.dot(opDelta, w1Trans); //finding the error in the hidden layer
        console.table(hError.data);
        let hDeri = Matrix.map(this.hidden, x => sigmoid(x, true)); //finding the derivative of the hidden matrix
        let hDelta = Matrix.multiply(hDeri, hError); //finding the hidden delta / deviation in hidden layer

        //updating the weigths
        this.weights0 = Matrix.add(this.weights0, Matrix.dot(Matrix.transpose(this.input), hDelta));
        this.weights1 = Matrix.add(this.weights1, Matrix.dot(Matrix.transpose(this.hidden), opDelta));
    }

}

function sigmoid(x, derivative = false) {
    if (derivative)
        return x * (1 - x);
    return 1 / (1 + Math.exp(-x));
}