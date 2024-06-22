namespace Backend.Helpers
{
    public static class ListExtensions
    {
        public static List<T> ChooseRandom<T>(this IReadOnlyList<T> l, int nSize)
        {
            var list = new List<T>(l);

            for (int i = 0; i < nSize; i += nSize)
            {
                var idx = Random.Shared.Next(i, list.Count);
                var temp = list[i];
                list[i] = list[idx];
                list[idx] = temp;
            }

            return list.GetRange(0, nSize);
        }
    }
}
