%source: http://stackoverflow.com/questions/14140746/how-to-find-the-index-of-the-n-smallest-elements-in-a-vector
function [smallestNElements smallestNIdx] = getNElements(A, n)
     [ASorted AIdx] = sort(A);
     smallestNElements = ASorted(1:n);
     smallestNIdx = AIdx(1:n);
end
