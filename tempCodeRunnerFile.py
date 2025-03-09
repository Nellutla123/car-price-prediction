        df['model'] = le.transform(df['model'])
        
        # Preprocess
        processed_data = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(processed_data)[0]
        
        return render_template('index.html', 
                            prediction=f"₹{round(prediction, 2):,}")
    
    except Exception as e: