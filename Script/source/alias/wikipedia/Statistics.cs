using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;
using pml.file.reader;
using pml.file.writer;

namespace msra.nlp.el.script.alias.wikipedia
{
    class Statistics
    {
        static Regex nonChineseRegex = new Regex(@"^[\u4e00-\u9fa5\s·]+$");

        public static int StatisticUniqueWordNumber(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            var set = new HashSet<char>();
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                line = line.Replace("\t","");
                var array = line.ToArray();
                foreach(var word in array)
                {
                    set.Add(word);
                }
            }
            foreach(var word in set)
            {
                writer.WriteLine(word);
            }
            reader.Close();
            writer.Close();
            return set.Count();
        }

    }
}
