# checks confusion_matrix and per_class_accuracy against one hand-computed example each. Good, since these have an exact right answer.

@testset "confusion_matrix" begin
    y  = ["normal", "normal", "pneumonia", "pneumonia", "pneumonia"]  # true
    yn  = ["normal", "pneumonia", "pneumonia", "pneumonia", "normal"]  # predicted

    cm = confusion_matrix(yn, y)  # cm[i,j] = predicted label i, true label j

    @test cm[1,1] == 1  # predicted normal,    true normal
    @test cm[2,1] == 1  # predicted pneumonia, true normal
    @test cm[1,2] == 1  # predicted normal,    true pneumonia
    @test cm[2,2] == 2  # predicted pneumonia, true pneumonia
    @test sum(cm) == length(y)
end

@testset "per_class_accuracy" begin
    cm = [8.0 2.0; 2.0 8.0]  # diag = correct, columns = true totals
    accs = per_class_accuracy(cm)
    @test accs ≈ [0.8, 0.8]
end