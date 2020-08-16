package com.dji.uxsdkdemo;

import android.Manifest;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import android.graphics.Path;
import android.graphics.SurfaceTexture;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import dji.common.error.DJIError;
import dji.common.useraccount.UserAccountState;
import dji.common.util.CommonCallbacks;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.useraccount.UserAccountManager;
import dji.ux.widget.FPVOverlayWidget;
import dji.ux.widget.FPVWidget;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;


import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.text.Text;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Timer;

import static java.lang.Thread.sleep;

public class MainActivity extends AppCompatActivity  {

    static {
        System.loadLibrary("opencv_java3");
    }


    private enum ButtonType {CLASSIFY, LIVESTREAM}
    private ButtonType buttonType = ButtonType.LIVESTREAM;
    Button bclassify;
    Button blive;

    //UI ELEMENTS
    TextView textResult;
    FPVWidget fpv;
    SurfaceView sview;
    SurfaceHolder sholder;

    //MISC - logs
    File file;
    ArrayList<Long> times =  new ArrayList<Long>();

    //TENSORFLOW
    private static final String MODEL_FILE = "file:///android_asset/model.pb";
    private static final String INPUT_NODE = "img";
    private static final String OUTPUT_NODE = "conv2d_19/Sigmoid";
    private static final int N_FEATURES = 512*512;
    private static final long[] INPUT_SIZE = {1,512,512,3};
    private TensorFlowInferenceInterface tensorflow = null;
    private String[] outputNodes = {OUTPUT_NODE};
    private float[] output = new float[N_FEATURES]; //converted from 2d to 1d
    private int[] intValues = new int[N_FEATURES];
    private float[] floatValues = new float[N_FEATURES*3];

    //SEGMENTATION RENDER THREAD
    private SegmentationThread segthread = null;
    private boolean hasSurface = false;
    private Paint alpha = new Paint();
    private boolean save = true;
    private long start_time;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // When the compile and target version is higher than 22, please request the
        // following permissions at runtime to ensure the
        // SDK work well.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.VIBRATE,
                            Manifest.permission.INTERNET, Manifest.permission.ACCESS_WIFI_STATE,
                            Manifest.permission.WAKE_LOCK, Manifest.permission.ACCESS_COARSE_LOCATION,
                            Manifest.permission.ACCESS_NETWORK_STATE, Manifest.permission.ACCESS_FINE_LOCATION,
                            Manifest.permission.CHANGE_WIFI_STATE, Manifest.permission.MOUNT_UNMOUNT_FILESYSTEMS,
                            Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.SYSTEM_ALERT_WINDOW,
                            Manifest.permission.READ_PHONE_STATE,
                    }
                    , 1);
        }

        setContentView(R.layout.activity_main);
        initUI();





    }

    @Override
    protected void onResume() {
        super.onResume();

        if(segthread !=null){
            segthread.mRunning = true;
        }


    }

    @Override
    protected  void onPause(){
        super.onPause();
        if(segthread != null){
            segthread.mRunning = false;
        }
    }

    private void initUI() {
        fpv = (FPVWidget) findViewById(R.id.fpv);
        textResult = (TextView) findViewById(R.id.result);
        sview = (SurfaceView)findViewById(R.id.sview);
        sview.setVisibility(View.INVISIBLE);
        bclassify = (Button)findViewById(R.id.classify);
        blive = (Button)findViewById(R.id.livestream);

        initPreviewer();

        blive.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(buttonType == ButtonType.CLASSIFY) { //prevent double selection
                    buttonType = ButtonType.LIVESTREAM;
                    if (segthread != null) {
                        segthread.stopRendering(); //turn off our segmentation algorithm
                    }

                    sview.setVisibility(View.INVISIBLE);
                    textResult.setText("Livestream");
                    //Log.v("LOG","LIVESTREAM");
                }

            }
        });

        bclassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(buttonType == buttonType.LIVESTREAM) { //prevent double selection
                    buttonType = ButtonType.CLASSIFY;

                    sview.setVisibility(View.VISIBLE);
                    Log.v("LOG","CLASSIFY");

                    if (segthread == null) {
                        segthread = new SegmentationThread(sholder);
                    }
                    segthread.mRunning=true;
                    try{wait(30);} //prevents FPV freezing until thread stuff is posted
                    catch (Exception e){}
                    segthread.start(); //turn on our segmentation algorithm
                    textResult.setText("Classify mode");

                }

            }
        });
        textResult.setText("Livestream");

        //for segmentation
        alpha.setAlpha(80);
        tensorflow = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        //for logs
        try{
            String root = Environment.getExternalStorageDirectory().toString();
            File direct = new File(String.format("%s/Timing_logs/",root));
           
            if (!direct.exists()) {
                direct.mkdir();
            }

            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String fileName = String.format("log_%s",timeStamp) + ".html";

            file = new File(String.format("%s/Timing_logs/",root) + File.separator + fileName);

            file.createNewFile();

        }
        catch (Exception e) {}

    }

    public void appendLog(long time){
        try {

            if (file.exists()) {
                times.add(time);
                //calc average
                int size = times.size();
                long total = 0;
                for(int i = 0; i < size; i++){
                    total = total+ times.get(i);
                }
                float avg = 1.0f * total / size;

                //output to log
                OutputStream fileOutputStream = new FileOutputStream(file, true);
                fileOutputStream.write(("<p style=\"background:lightgray;\"><strong style=\"background:lightpink;\">&nbsp&nbsp" + "timinglog" + " :&nbsp&nbsp</strong>&nbsp&nbsp" + String.format("%d",time) + "<strong style=\"background:lightpink;\">&nbsp&nbsp" + "averagelog" + " :&nbsp&nbsp</strong>&nbsp&nbsp" + String.format("%.2f",avg)+"</p>").getBytes() );
            
                fileOutputStream.close();


            }


            

        } catch (Exception e) {
            //Log.e(TAG, "Error while logging into file : " + e);
        }
    }


    private void SaveImage(Bitmap bitmap, String image_name){
      
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(String.format("%s/Results/",root));
        myDir.mkdirs();
        String fname = "Image-" + image_name+ ".jpg";
        File file = new File(myDir, fname);
        if (file.exists()) file.delete();
        Log.i("LOAD", root + fname);
        try {
            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    private void showToast(String s) {
        Toast.makeText(getApplicationContext(), s, Toast.LENGTH_SHORT).show();
    }

    private void initPreviewer() {
        Log.v("LOG","initPreviewSurfaceView");
        sholder = sview.getHolder();
        //segthread = new SegmentationThread(sholder);
        sholder.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                Log.d("LOG", "real onSurfaceCreated");
                hasSurface = true;
                /*if(segthread != null){
                    segthread.start();
                }*/
                //get surface holder surface
                //sholder.getSurface();
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                Log.d("LOG", "real onSurfaceChanged");
                //sholder.getSurface();
                //sholder = sview.getHolder();


            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                Log.d("LOG", "real onSurfaceDestroyed");
                if(segthread!=null){
                    segthread.stopRendering();
                    hasSurface = false;
                    /*boolean retry = true;
                    while (retry) {
                        try {
                            segthread.join(); //join main thread
                            retry = false;
                        } catch (InterruptedException e) { }
                    }*/

                }

            }
        });
    }

    private void computeoverlay(Bitmap bitmap, Canvas c){
        
        //Log.v("LOG","compute overlay");
        //long start_time = System.currentTimeMillis();
        int width = bitmap.getWidth(); //2048
        int height = bitmap.getHeight(); //1488

        //Pre-process image
       
        Mat ImageMat = new Mat();        
        Utils.bitmapToMat(bitmap, ImageMat);

        //crop to aspect ratio 1:1
        //Log.v("LOG",String.format("width %d  height: %d",width,height));
        Rect ROI = new Rect(280,0,height,height); //margin is 500 if the image is the same size as it was in training 3000,4000 //DONT FORGET TO UNCOMMENT!!!!!!!!!!!!
        ImageMat =ImageMat.submat(ROI); //I uncommented these
        //Log.v("LOG",String.format("Raw bitmap %d %d", ImageMat.height(), ImageMat.width()));

        //get pixel values and make prediction
        Imgproc.resize(ImageMat, ImageMat, new Size(512, 512)); //Cv_U8=0
        Bitmap bmp3 = Bitmap.createBitmap(512,512,Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(ImageMat, bmp3);
        bmp3.getPixels(intValues, 0, bmp3.getWidth(), 0, 0, bmp3.getWidth(), bmp3.getHeight());

        for (int i = 0; i < intValues.length; i++) {
            // Colors are represented as packed ints, made up of 4 bytes: alpha, red, green, blue. The values are unpremultiplied, meaning any transparency is stored solely in the alpha component, and not in the color components.
            // The components are stored as follows (alpha << 24) | (red << 16) | (green << 8) | blue.
            int val = intValues[i];

            final int R = Color.red(val);//(val << 16) & 0xff;
            final int G = Color.green(val);//(val << 8) & 0xff;
            final int B = Color.blue(val);//val & 0xff;

            floatValues[i*3] = B/255.0f;
            floatValues[i*3+1] = G/255.0f;
            floatValues[i*3+2] = R/255.0f;
            
        }

        //apply inference
        //tensorflow = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        tensorflow.feed(INPUT_NODE,floatValues,INPUT_SIZE);
        tensorflow.run(outputNodes, false);
        tensorflow.fetch(OUTPUT_NODE, output);

        //threshold output nodes to get classes and create output bitmap
        Bitmap result = Bitmap.createBitmap(512,512,Bitmap.Config.ARGB_8888);
        for(int y=0; y < 512; y++){
            for(int x = 0; x< 512;x++){
                
                int purple = Color.rgb(75,0,130);

                result.setPixel(x,y,output[y*512+x] <=0.5 ? purple : Color.YELLOW);
            }
        }

        //prepare result for display

        //convert prediction from 512 x 512 array to overlay shape
        int w = sview.getWidth();
        final Bitmap resized = Bitmap.createScaledBitmap(bmp3, w, w, true); //height and width of display window
        c.drawBitmap(resized,0,0,null); //draw over the central ROI

        //draw over the central ROI
        final Bitmap resized2 = Bitmap.createScaledBitmap(result, w, w, true);    //overlay output over existing fpv
        c.drawBitmap(resized2,0,0,alpha);

        long end_time = System.currentTimeMillis();
        long difference = end_time-start_time;
        Log.v("LOG",String.format("Compute overlay time: %d ms",difference));
        appendLog(difference);

        //display and save to internal storage
        Thread saver = new Thread(new Runnable() {
            @Override
            public void run() {

                Bitmap overlay = Bitmap.createBitmap(resized.getWidth(), resized.getHeight(), resized.getConfig());
                Canvas canvas = new Canvas(overlay);
                canvas.drawBitmap(resized, 0, 0, null);
                canvas.drawBitmap(resized2, 0, 0, alpha);
                

                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                SaveImage(overlay, String.format("%s_output", timeStamp));
                SaveImage(resized, String.format("%s_original", timeStamp));
                
                
            }
        });
        saver.start();





    }

    private class SegmentationThread extends Thread{
        private SurfaceHolder mHolder;
        private volatile boolean mRunning = true;

        public SegmentationThread(SurfaceHolder surfaceholder) {

            mHolder = surfaceholder;

        }

        void stopRendering() {
            interrupt();
            mRunning = false;
            try {
                join();
            } catch (InterruptedException ignored) {
            }
        }

        @Override
        public void run() {
            Log.v("LOG","Thread running");
          
            while (mRunning && !Thread.interrupted()) {

                Canvas canvas = null;
                mHolder = sview.getHolder();
                try{
                    canvas = mHolder.lockCanvas(null);
                    start_time = System.currentTimeMillis();
                    Bitmap bitmap = fpv.getBitmap();    //get the first person view
                    

                    if(bitmap!=null && canvas !=null){
                       
                        computeoverlay(bitmap,canvas);      //perform segmentation and render result

                    }
                   
                }finally {
                    if(canvas != null){
                        mHolder.unlockCanvasAndPost(canvas);
                        

                    }
                }
                //To get a constant updating of the seg result - the next two lines should be commented out.
                //Temporarily uncommented to only get the seg result on button click 
                mRunning=false;
                buttonType = buttonType.LIVESTREAM;



            }

        }


    }

}

