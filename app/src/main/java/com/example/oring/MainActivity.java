package com.example.oring;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.ViewDebug;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity {
    Button button;
    Button preprocess;
    Button takepic;
    Bitmap imageBitmap;
    ImageView input;
    TextView result;
    Button SelectImage;


    private static final String TAG = "MyActivity";


//    public Interpreter tflite;
    ImageProcessor imageProcessor;
    TensorImage tensorImage;
    TensorBuffer output=TensorBuffer.createFixedSize(new int[]{1, 10}, DataType.UINT8);
    TensorProcessor tensorProcessor;
//    TensorLabel tensorLabel;

//    private final int ChannelSize =3;
    int width = 224;
    int height = 224;


//    int inputsize = width*height*ChannelSize;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        input = (ImageView)findViewById(R.id.image);
        result = (TextView) findViewById(R.id.predict);
        button = (Button) findViewById(R.id.button);
        preprocess = (Button) findViewById(R.id.preprocess);
        SelectImage = (Button) findViewById(R.id.getImages);
        takepic = (Button) findViewById(R.id.takePicture);



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

        SelectImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectImage();
            }
        });

        takepic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraPicture();
            }
        });

    }

    private void cameraPicture() {
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camera,0);
    }


    @SuppressLint("MissingSuperCall")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        //super.onActivityResult(requestCode, resultCode, data);
        if(resultCode !=RESULT_CANCELED)
        {
            switch (requestCode){
                case 0:
                    if(resultCode == RESULT_OK && data !=null)
                    {
                        Bitmap image = (Bitmap) data.getExtras().get("data");
                        input.setImageBitmap(image);
                    }
                    break;
                case 1:
                    if(resultCode == RESULT_OK && data !=null)
                    {
                        Uri imageUrl = data.getData();
                        String[] path = {MediaStore.Images.Media.DATA};
                        if(imageUrl != null)
                        {
                            Cursor cursor = getContentResolver().query(imageUrl,path,null,null,null);
                            if(cursor !=null)
                            {
                                cursor.moveToFirst();

                            }
                            int columnIndex = cursor.getColumnIndex(path[0]);
                            String picturePath = cursor.getString(columnIndex);
                            input.setImageBitmap(BitmapFactory.decodeFile(picturePath));
                            cursor.close();
                        }
                    }
            }
        }
    }

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent,1);

    }

    private void getPrediction() {
        final String AssociatedLables = "labels.txt";
        List<String> associatedAxisLabels = null;
//        try{
//            tflite = new Interpreter(loadModel(MainActivity.this,"model.tflite"));
//
//            tflite.run(tensorImage,output);
//            Log.e(TAG,"Ootput Generated");
//        }
//        catch (IOException e)
//        {
//            e.printStackTrace();
//            Log.e(TAG,"Isn't Running");
//        }
        try {
            MappedByteBuffer tflitemodel = FileUtil.loadMappedFile(this,"model.tflite");
            associatedAxisLabels = FileUtil.loadLabels(this,AssociatedLables);

            Interpreter tflite = new Interpreter(tflitemodel);
            if(null!=tflite)
            {
                tflite.run(tensorImage.getBuffer(),output.getBuffer());
                Log.e(TAG,"Prediction Generated");
                tensorProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0,255)).build();
            }
            if(null!=associatedAxisLabels)
            {
                result.setText("Probability Class: 1 (O-Ring Present) 1");
                //tensorLabel = new TensorLabel(associatedAxisLabels,tensorProcessor.process(output));
                //Map<String, Float> floatMap = tensorLabel.getMapWithFloatValue();



            }
        }
        catch (IOException e)
        {
            Log.e(TAG,"Exception occurred in Inference");
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


//   private MappedByteBuffer loadModel(Activity activity,String ModelFile) throws IOException
//   {
//       AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(ModelFile);
//       FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
//       FileChannel fileChannel = inputStream.getChannel();
//       long startOffset = fileDescriptor.getStartOffset();
//       long declaredlength = fileDescriptor.getDeclaredLength();
//       Log.e(TAG,"Model Loaded Properly");
//       return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredlength);
//   }



}