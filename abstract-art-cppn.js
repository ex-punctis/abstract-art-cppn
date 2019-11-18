//v1

class abstractANN {
    constructor({canvasID, saveButtonID,  resetButtonID, activationIDs,  magnitudeIDs }={}) {

        tf.enableProdMode();
        //console.log(tf.getBackend());

        // DOM elments
        this.canvas = document.getElementById(canvasID);
        // buttons
        this.resetButton = document.getElementById(resetButtonID)
        this.saveButton = document.getElementById(saveButtonID);
        // drop-down lists for activation functions
        this.activationDropDown1 = document.getElementById(activationIDs[0]);
        this.activationDropDown2 = document.getElementById(activationIDs[1]);
        this.activationDropDown3 = document.getElementById(activationIDs[2]);
        this.activationDropDown4 = document.getElementById(activationIDs[3]);
        // magnitudes sliders 
        this.magnitudeSlider1 = document.getElementById(magnitudeIDs[0]);
        this.magnitudeSlider2 = document.getElementById(magnitudeIDs[1]);
        this.magnitudeSlider3 = document.getElementById(magnitudeIDs[2]);
        

        
        // activation functions
        this.ACTIVATION = {
            'cos': (input) => input.cos(),
            'sin': (input) => input.sin(),
            'tanh': (input) => input.tanh(),
            'lin': (input) => input
        };
        this.activation1 = this.ACTIVATION[this.activationDropDown1.value];
        this.activation2 = this.ACTIVATION[this.activationDropDown2.value];
        this.activation3 = this.ACTIVATION[this.activationDropDown3.value];
        this.activation4 = this.ACTIVATION[this.activationDropDown4.value];

        // post-activation magnitudes
        this.magnitude1 = tf.scalar(this.magnitudeSlider1.value, 'float32');
        this.magnitude2 = tf.scalar(this.magnitudeSlider2.value, 'float32');
        this.magnitude3 = tf.scalar(this.magnitudeSlider3.value, 'float32');


        // constants
        this.WIDTH = 100;   // linear resolution
        this.H = 8;         // number of neurons in hidden layers
        
        // canvas element
        //this.canvas.style.height = "60vh";
        this.canvasPos = this.canvas.getBoundingClientRect();

        // mouse and trackpad input processing
        this.mouseX = 0;
        this.mouseY = 0;
        // attach mousemove and touch move events
        this.canvas.addEventListener("mousemove", 
            (event) => {
                event.preventDefault();
                this.mouseX = (event.clientX - this.canvasPos.left)/300;
                this.mouseY = (event.clientY - this.canvasPos.top)/300;
            }
        );
        this.canvas.addEventListener('touchmove', 
            (event) => {
                event.preventDefault();
                this.mouseX = (event.touches[0].clientX - this.canvasPos.left)/300;
                this.mouseY = (event.touches[0].clientY - this.canvasPos.top)/300;
            }
        );	
        
        // attach drop-down onchange events
        this.activationDropDown1.addEventListener("change", event => { this.activation1 = this.ACTIVATION[this.activationDropDown1.value] } );
        this.activationDropDown2.addEventListener("change", event => { this.activation2 = this.ACTIVATION[this.activationDropDown2.value] } );
        this.activationDropDown3.addEventListener("change", event => { this.activation3 = this.ACTIVATION[this.activationDropDown3.value] } );
        this.activationDropDown4.addEventListener("change", event => { this.activation4 = this.ACTIVATION[this.activationDropDown4.value] } );
        
        // attach slider input events
        this.magnitudeSlider1.addEventListener("input", event => { 
            this.magnitude1.dispose();
            this.magnitude1 = tf.scalar(this.magnitudeSlider1.value, 'float32'); 
        } );
        this.magnitudeSlider2.addEventListener("input", event => { 
            this.magnitude2.dispose();
            this.magnitude2 = tf.scalar(this.magnitudeSlider2.value, 'float32'); 
        } );
        this.magnitudeSlider3.addEventListener("input", event => { 
            this.magnitude3.dispose();
            this.magnitude3 = tf.scalar(this.magnitudeSlider3.value, 'float32'); 
        } );


        // input and weights
        this.input = [];
        this.inputTensorXY;
        this.w1;
        this.w2;
        this.w3;
        this.w4;


        // temporal inputs
        this.t = 0;
        // temporal input steps (determines frequency)
        this.tSTEP = Math.PI/800;
        // temoral componenent magnitude
        this.tMagnitude = tf.scalar(1.2);
        // layer magnitude
        //this.lMagnitude = tf.scalar(1);

        // create hidden canvas to generate wallpaper and convert to image
        this.HIDDENWIDTH = 1920;
        this.HIDDENHEIGHT = 1080;
        this.hiddenCanvas = document.createElement('canvas');
        this.hiddenCanvas.width = this.HIDDENWIDTH;
        this.hiddenCanvas.height = this.HIDDENHEIGHT;
        this.ctx = this.hiddenCanvas.getContext('2d');
        this.hiddenLink = document.createElement('a');
        
        // the wallpaper is pretty high-res, so we'll generate the image batch-by-batch
        // number of batches
        this.numBatches =  10;

        // capture "space" key press
        document.onkeydown = (event) => {
            event = event || window.event;
            event.preventDefault();
            if (event.code == 'Space') { 
                this.saveHighResFrame();
            }
        };
        // wallpaper button
        this.saveButton.onclick = _ => this.saveHighResFrame();

        // button to reset weights
        this.resetButton.onclick = _ => {
            this.stop();
            setTimeout(() => {
                this.inputTensorXY.dispose();
                this.w1.dispose();
                this.w2.dispose();
                this.w3.dispose();
                this.w4.dispose();
            }, 50); // waiting helps prevent an error

            setTimeout(() => { this.start(); }, 50); //must wait or else there will be a memory leak
        
        };

        // run flag
        this.runFlag = false;

    }

    generateInputs() {
        this.input = [];
        // generate matrix of all (x,y) combinations
        for (let i = 0; i<this.WIDTH; i++) {
            for (let j = 0; j<this.WIDTH; j++) {
                this.input.push([i/this.WIDTH, j/this.WIDTH]); //i*j/width/width
            }
        }
        this.inputTensorXY = tf.tensor2d(this.input, [this.WIDTH*this.WIDTH, 2],  'float32');
    }

    generateWeights() {
        // initialize random weights
        this.w1 = tf.randomNormal([5, this.H]); // x, y + mouse.x, mouse.y + 1 temporal input
        this.w2 = tf.randomNormal([this.H, this.H]);
        this.w3 = tf.randomNormal([this.H, this.H]);
        this.w4 = tf.randomNormal([this.H, 3]); // 3 outputs (rgb)
    }


    start() {
        this.generateInputs();
        this.generateWeights();
        this.runFlag = true;
        window.requestAnimationFrame(() => { this.drawFrame(); });
    }

    continue() {
        this.runFlag = true;
        window.requestAnimationFrame(() => { this.drawFrame(); });
    }

    stop() {
        this.runFlag = false;
    }


    // Method to perform claculations, draw a frame, make a recursive call to draw next frame
    async drawFrame() {

        let outputTensorRGB = tf.tidy(() => {            
            // add sine of temporal inputs, broadcast along the first axis, concatenate with input tensor
            let inputTensorXYT = tf.concat( 
                [ 
                    this.inputTensorXY, 
                    tf.tensor2d([this.mouseX, this.mouseY, Math.sin(this.t)], [1,3]).mul(this.tMagnitude).tile([this.WIDTH*this.WIDTH,1]) 
                ], 1 
            );
            
            // perform forward propagation and return rgb result
            let z1 = inputTensorXYT.matMul(this.w1);
            let a1 = this.activation1(z1).mul(this.magnitude1);
            let z2 = a1.matMul(this.w2);
            let a2 = this.activation2(z2).mul(this.magnitude2);
            let z3 = a2.matMul(this.w3);
            let a3 = this.activation3(z3).mul(this.magnitude3);
            let z4 = a3.matMul(this.w4);
            let a4 = this.activation4(z4);
            let output = a4.div(tf.scalar(2)).add(tf.scalar(0.5)).reshape([this.WIDTH, this.WIDTH, 3]);

            return (output);
        });

        // visualize frame and dispose output
        await tf.browser.toPixels(outputTensorRGB, canvas); 
        outputTensorRGB.dispose();

        // call this method recursively to generate next frame
        if(this.runFlag){
            await tf.nextFrame();
            
            this.t = this.t+this.tSTEP;
            if (this.t > 6.283185308) {
                this.t = this.t - 2*Math.PI; 
            }

            window.requestAnimationFrame(() => { this.drawFrame(); });
        }
    }


    // Method to generate higher resolution wallpaper
    saveHighResFrame() {
        let outputHighRes = [];
        this.runFlag = false;

        // // generate matrix of all (x,y) combinations
        for (let batch = 0; batch < this.HIDDENHEIGHT/this.numBatches; batch++) {
            let row = [];
            for (let i = 0; i<this.numBatches; i++) { 
                for (let j = 0; j<this.HIDDENWIDTH; j++) {
                    row.push([(batch*this.numBatches+i)/this.HIDDENHEIGHT, j/this.HIDDENHEIGHT]); //i*j/width/width
                }
            }
            let outputRowRGB = tf.tidy(() => {
                // add sine of temporal inputs, broadcast along the first axis, concatenate with input tensor
                let inputRowTensor = tf.concat( [ 
                    tf.tensor2d(row, [this.HIDDENWIDTH*this.numBatches, 2],  'float32'), 
                    tf.tensor2d([this.mouseX, this.mouseY, Math.sin(this.t)], [1,3]).mul(this.tMagnitude).tile([this.HIDDENWIDTH*this.numBatches,1]) 
                ], 1 );

                // perform forward propagation and return rgb result
                let z1 = inputRowTensor.matMul(this.w1);
                let a1 = this.activation1(z1).mul(this.magnitude1);
                let z2 = a1.matMul(this.w2);
                let a2 = this.activation2(z2).mul(this.magnitude2);
                let z3 = a2.matMul(this.w3);
                let a3 = this.activation3(z3).mul(this.magnitude3);
                let z4 = a3.matMul(this.w4);
                let a4 = this.activation4(z4);
                let output = a4.mul(tf.scalar(127.5)).add(tf.scalar(127.5)).floor();

                return (tf.concat( [output, tf.tensor2d([255], [1,1]).tile([this.HIDDENWIDTH*this.numBatches,1]) ], 1 ));
            } ); 

            let tempArray = outputRowRGB.dataSync();
            outputRowRGB.dispose();
            // outputHighRes.push(...tempArray); // doesn't work on ipad (call stack issues)
            for (let i=0; i<tempArray.length; i++) {    outputHighRes.push(tempArray[i]);   }
        }

        let idata = this.ctx.createImageData(this.HIDDENWIDTH, this.HIDDENHEIGHT);
        idata.data.set(Uint8ClampedArray.from(outputHighRes));
        this.ctx.putImageData(idata, 0, 0);

        // // create saveable link
        // this.hiddenLink.setAttribute('download', 'art.jpg');
        // this.hiddenLink.setAttribute('href', this.hiddenCanvas.toDataURL("image/jpeg").replace("image/jpeg", "image/octet-stream"));
        // this.hiddenLink.click();

        // display generated image as saveable thumbnail
        let image = document.createElement('img');
        image.src = this.hiddenCanvas.toDataURL("image/jpeg");
        image.style.width = this.HIDDENWIDTH/10;
        image.style.height = this.HIDDENHEIGHT/10;
        document.body.appendChild(image);
        
        // resume running
        this.runFlag = true;
    }

    // saveHighResFrames() {
    //     for (let i=0; i<200; i++ ) {
    //         this.t = this.t+this.tSTEP
    //         this.saveHighResFrame();
    //     }
    // }
}