 "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hnYHKGEsPR3b",
        "outputId": "a6481bc6-7590-419c-bdb1-134c8b0d7adc"
      },
      "source": [
        "model=tuner_search.get_best_models(num_models=1)[0]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter\n",
            "WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1\n",
            "WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2\n",
            "WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay\n",
            "WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate\n",
            "WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to mak
