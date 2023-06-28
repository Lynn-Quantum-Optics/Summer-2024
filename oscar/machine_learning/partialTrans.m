function Y = partialTrans(X)
    X([5 2]) = X([2 5]);
    X([4 7]) = X([7 4]);
    X([10 13]) = X([13 10]);
    X([12 15]) = X([15 12]);
    Y = X;
end