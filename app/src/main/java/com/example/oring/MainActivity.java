package com.example.oring;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

public class MainActivity extends AppCompatActivity {
    Button button;
    Button preprocess;
    Bitmap imageBitmap;
    ImageView input;
    TextView result;


    private static final String TAG = "MyActivity";


    public Interpreter tflite;
    ImageProcessor imageProcessor;
    TensorImage tensorImage;

    private int ChannelSize =3;
    int width = 224;
    int height = 224;

    String output;

    int inputsize = width*height*ChannelSize;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        input = (ImageView)findViewById(R.id.image);
        result = (TextView) findViewById(R.id.predict);
        button = (Button) findViewById(R.id.button);
        preprocess = (Button) findViewById(R.id.preprocess);





        preprocess.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getBitmap();
            }
        });
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
              getPrediction();
            }
        });

    }

    private void getPrediction() {
        try{
            tflite = new Interpreter(loadModel(MainActivity.this,"model.tflite"));

            tflite.run(tensorImage,output);
            Log.e(TAG,"Ootput Generated");
        }
        catch (IOException e)
        {
            e.printStackTrace();
            Log.e(TAG,"Isn't Running");
        }
    }


    public void getBitmap()
    {
        BitmapDrawable bitmapDrawable = (BitmapDrawable) input.getDrawable();
        imageBitmap = (Bitmap) bitmapDrawable.getBitmap();

        imageBitmap = Bitmap.createScaledBitmap(imageBitmap,height,width,false);
        Log.e(TAG,"Bitmap Created Successfully");
        input.setImageBitmap(imageBitmap);

        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224,224, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        tensorImage = new TensorImage(DataType.UINT8);
        tensorImage.load(imageBitmap);
        tensorImage = imageProcessor.process(tensorImage);


    }

   private MappedByteBuffer loadModel(Activity activity,String ModelFile) throws IOException
   {
       AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(ModelFile);
       FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
       FileChannel fileChannel = inputStream.getChannel();
       long startOffset = fileDescriptor.getStartOffset();
       long declaredlength = fileDescriptor.getDeclaredLength();
       Log.e(TAG,"Model Loaded Properly");
       return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredlength);
   }



}