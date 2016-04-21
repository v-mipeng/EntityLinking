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
    class DataFilter
    {
        static Regex nonChineseRegex = new Regex(@"^[\u4e00-\u9fa5\s·]+$");


        public static void FilterOutNonChineseItem(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;

            while((line = reader.ReadLine())!=null)
            {
                if (nonChineseRegex.IsMatch(line))
                {
                    writer.WriteLine(line);
                }
            }
            reader.Close();
            writer.Close();
        }

        public static void PruneRedirect(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            var deleteEnglishRegex = new Regex(@"^[\u4e00-\u9fa5\s·]+(\d+)?$");
            string line;
            var set = new HashSet<string>();

            while ((line = reader.ReadLine()) != null)
            {
                set.Clear();
                var array = line.Split('\t');
                if(ContainNonChineseCharacter(array[0]))
                {
                    continue;
                }
                for(var i = 1;i<array.Length;i++)
                {
                    if(!ContainNonChineseCharacter(array[i]))
                    {
                        set.Add(array[i]);
                    }
                }
                writer.Write(array[0]);
                foreach(var redirect in set)
                {
                    writer.Write("\t" + redirect);
                }
                writer.WriteLine("");
            }
            reader.Close();
            writer.Close();
        }

        /// <summary>
        /// Combine anchor-->entity
        /// </summary>
        /// <param name="source"></param>
        /// <param name="des"></param>
        public static void CombineAnchor(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;
            var dic = new Dictionary<string, HashSet<string>>();

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                try
                {
                    var set = dic[array[0]];
                    set.Add(array[1]);
                }
                catch(Exception)
                {
                    var set = new HashSet<string>();
                    set.Add(array[1]);
                    dic[array[0]] = set;
                }
            }
            reader.Close();
            foreach(var pair in dic)
            {
                writer.Write(pair.Key);
                foreach(var entity in pair.Value)
                {
                    writer.Write("\t" + entity);
                }
                writer.WriteLine("");
            }
            writer.Close();
        }

        public static void ConstructEntityToAnchors(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;
            var dic = new Dictionary<string, HashSet<string>>();

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                try
                {
                    var set = dic[array[1]];
                    set.Add(array[0]);
                }
                catch (Exception)
                {
                    var set = new HashSet<string>();
                    set.Add(array[0]);
                    dic[array[1]] = set;
                }
            }
            reader.Close();
            foreach (var pair in dic)
            {
                writer.Write(pair.Key);
                foreach (var entity in pair.Value)
                {
                    writer.Write("\t" + entity);
                }
                writer.WriteLine("");
            }
            writer.Close();
        }

        public static bool ContainNonChineseCharacter(string input)
        {
            if (input == null)
            {
                throw new ArgumentNullException("The input should not be null!");
            }
            if (nonChineseRegex.IsMatch(input))
            {
                return false;
            }
            else
            {
                return true;
            }

        }

        public static void FilterOutAnchorWithSameEntity(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                if(array.Length!=2 || array[0].Equals(array[1]))
                {
                    continue;
                }
                writer.WriteLine(line);
            }
            reader.Close();
            writer.Close();
        }

        public static void ConstructEntityToMention(string source, string des)
        {
            var reader = new LargeFileReader(source);
            var writer = new LargeFileWriter(des, FileMode.Create);
            var dic = new Dictionary<string, HashSet<String>>();
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                var array = line.Split('\t');
                try
                {
                    var set = dic[array[0]];
                    for (var i = 1; i < array.Length; i++)
                    {
                        if (!array[i].Equals(array[0]))
                        {
                            set.Add(array[i]);
                        }
                    }
                }
                catch (Exception)
                {
                    var set = new HashSet<string>();
                    for (var i = 1; i < array.Length; i++)
                    {
                        if (!array[i].Equals(array[0]))
                        {
                            set.Add(array[i]);
                        }
                    }
                    dic[array[0]] = set;
                }
            }
            reader.Close();
            foreach (var pair in dic)
            {
                writer.Write(pair.Key);
                foreach (var entity in pair.Value)
                {
                    writer.Write("\t" + entity);
                }
                writer.WriteLine("");
            }
            writer.Close();
        }
    }
}
